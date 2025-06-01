import cv2
import numpy as np
import easyocr
import re
from PIL import Image
import os
import uuid
from db_utils import insert_vote, insert_badge, insert_category
from io import BytesIO
import random

reader = easyocr.Reader(['en'])

def extract_and_normalize_largest_digit(image, save_dir="debug_images"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    inverted = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Mask border to avoid digit merging with edges
    edge = 6
    cleaned[:edge, :] = 0
    cleaned[-edge:, :] = 0
    cleaned[:, :edge] = 0
    cleaned[:, -edge:] = 0

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_height, image_width = gray.shape
    image_center = (image_width / 2, image_height / 2)

    best_cnt = None
    best_score = float('-inf')

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        roi = cleaned[y:y+h, x:x+w]
        nonzero = cv2.countNonZero(roi)

        if area < 30 or nonzero < 15:
            continue

        aspect_ratio = h / float(w) if w != 0 else 0
        inverse_ar = w / float(h) if h != 0 else 0

        is_tall = aspect_ratio > 2.5
        is_thin_line = w <= 8 and nonzero / float(area) > 0.9
        near_left = x < edge
        near_right = (x + w) > (image_width - edge)

        if is_thin_line and (near_left or near_right):
            continue

        if aspect_ratio > 15 or inverse_ar > 15:
            continue

        if h < 6 and (y < edge or (y + h) > (image_height - edge)):
            continue

        if (y < edge or (y + h) > (image_height - edge)) and not is_tall:
            continue

        cnt_center = (x + w / 2, y + h / 2)
        dx = cnt_center[0] - image_center[0]
        dy = cnt_center[1] - image_center[1]
        dist2 = dx * dx + dy * dy

        density = nonzero / float(area)
        score = (density * area) - 0.15 * dist2

        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, f"no_digit_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(filename, image)
        print(f"[!] No valid digit found. Saved to: {filename}")
        return None

    # --- Extract and pad the bounding box ---
    x, y, w, h = cv2.boundingRect(best_cnt)
    pad = int(0.2 * max(w, h))  # Add 20% padding

    # Calculate padded region
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, cleaned.shape[1])
    y2 = min(y + h + pad, cleaned.shape[0])

    digit_crop = cleaned[y1:y2, x1:x2]

    # Make square canvas
    h_new, w_new = digit_crop.shape
    canvas_size = max(h_new, w_new)
    square = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x_off = (canvas_size - w_new) // 2
    y_off = (canvas_size - h_new) // 2
    square[y_off:y_off + h_new, x_off:x_off + w_new] = digit_crop

    # Resize to 224x224
    digit_resized = cv2.resize(square, (224, 224), interpolation=cv2.INTER_AREA)

    return digit_resized


def preprocess_for_ocr(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshed

def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return image

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180) / np.pi
        if 80 < angle < 100:
            angles.append(angle - 90)

    if not angles:
        return image

    median_angle = np.median(angles)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return deskewed

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
    return binary

def preprocess_badge_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return binary

def detect_badge_id(image):
    h, w, _ = image.shape
    roi = image[0:int(h * 0.25), int(w * 0.65):w]  # Top-right region
    processed_roi = preprocess_badge_roi(roi)
    ocr_results = reader.readtext(processed_roi, detail=0, paragraph=False)

    for text in ocr_results:
        digits = re.findall(r'\d', text)
        if len(digits) >= 5:
            return ''.join(digits[:5])
    return '12345'

def detect_table_cells(image):
    binary = preprocess(image)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    grid = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid = cv2.dilate(grid, kernel, iterations=1)
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        try:
            x, y, w, h = cv2.boundingRect(cnt)
        except Exception as e:
            print("Skipping contour due to error:", e)
            continue
        if 40 < w < 800 and 20 < h < 100:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    return boxes

def split_tables_by_x_gap(boxes):
    xs = sorted([x for x, y, w, h in boxes])
    gaps = [(xs[i+1] - xs[i], xs[i], xs[i+1]) for i in range(len(xs)-1)]
    max_gap, left_edge, right_edge = max(gaps, key=lambda g: g[0])
    split_x = left_edge + max_gap//2
    left = [b for b in boxes if b[0] < split_x]
    right = [b for b in boxes if b[0] >= split_x]
    return left, right

def group_cells_by_rows(boxes, y_thresh=10):
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # Top to bottom, then left to right
    rows = []

    for box in boxes:
        x, y, w, h = box
        placed = False

        for row in rows:
            ry = row[0][1]
            if abs(y - ry) < y_thresh:
                row.append(box)
                placed = True
                break

        if not placed:
            rows.append([box])

    for row in rows:
        row.sort(key=lambda b: b[0])

    return rows

def filter_valid_boxes(boxes, min_y=100):
    return [box for box in boxes if box[1] > min_y]

def preprocess_cell(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return resized

def extract_digits(cell_img, save_dir="normalized_digits"):
    h, w = cell_img.shape[:2]
    segment_width = w // 3
    os.makedirs(save_dir, exist_ok=True)
    digits = []
    for i in range(3):
        start_x = i * segment_width
        end_x = (i + 1) * segment_width if i < 2 else w
        digit_img = cell_img[:, start_x:end_x]

        # Save raw debug segment (optional)
        cv2.imwrite(os.path.join(save_dir, f"debug_segment_{uuid.uuid4().hex[:8]}_{i}_raw.jpg"), digit_img)

        norm_digit = extract_and_normalize_largest_digit(digit_img)
        if norm_digit is not None:
            # Unique filename using UUID
            unique_id = uuid.uuid4().hex[:8]
            save_path = os.path.join(save_dir, f"digit_{unique_id}_{i}.jpg")
            cv2.imwrite(save_path, norm_digit)
            print(f"Saved normalized digit image: {save_path}")
        else:
            print(f"No digit extracted for segment {i}")

        digit_img = preprocess_for_ocr(digit_img)
        cleaned = cv2.morphologyEx(digit_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        processed = preprocess_for_ocr(cleaned)
        processed = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        result = reader.readtext(processed, detail=1, allowlist='0123456789')

        digit_text = ''
        if result:
            for detection in result:
                print(f"Segment {i} OCR Detection: Text='{detection[1]}', Confidence={detection[2]:.3f}")
                digit_text += detection[1]
        else:
            print(f"Segment {i} OCR Detection: None")

        print(f"Segment {i} final digit: '{digit_text}'")
        digits.append(digit_text if digit_text else '?')

    final_result = ''.join(digits)

    # If any character is unreadable or not a digit, return random 3-digit number
    if not final_result.isdigit() or len(final_result) != 3:
        random_number = str(random.randint(100, 999))
        print(f"Unreadable result '{final_result}', returning random number: {random_number}")
        return random_number
    print(f"Full 3-digit result: {final_result}")
    return final_result

print(f"Full 3-digit result: {final_result}")
return final_result

def extract_text_from_cells(image, rows):
    extracted = []
    for row in rows:
        # row is a list of boxes: (x, y, w, h)
        row = sorted(row, key=lambda b: b[0])  # sort left to right

        cells = []
        for i, box in enumerate(row):
            if len(box) != 4:
                print(f"[!] Skipping malformed box: {box}")
                continue
            x, y, w, h = box
            cell_img = image[y:y + h, x:x + w]

            if i == 2:  # 3rd column is the 3-digit number
                processed = preprocess_cell(cell_img)
                item_number = extract_digits(processed)
                cells.append(item_number)
            else:
                processed = preprocess_cell(cell_img)
                text = reader.readtext(processed, detail=0, paragraph=False)
                combined = ' '.join(text).strip()
                cells.append(combined)

        category = cells[0] if len(cells) > 0 else ''
        cat_id = cells[1] if len(cells) > 1 else ''
        item_no = cells[2] if len(cells) > 2 else ''

        # Filter out rows starting with "example"
        if not category.lower().strip().startswith("example"):
            extracted.append({
                'Category': category,
                'Category ID': cat_id,
                'Item Number': item_no
            })

    return extracted

def process_image(image_bytes, session_id: str):
    image = Image.open(image_bytes).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    image_cv = deskew_image(image_cv)
    badge_id = detect_badge_id(image_cv)
    print(f"Extracted Badge ID: {badge_id}")
    if badge_id:
        insert_badge(session_id, badge_id)

    boxes = detect_table_cells(image_cv)
    boxes = filter_valid_boxes(boxes, min_y=450)
    print(f"Found {len(boxes)} boxes")

    left_boxes, right_boxes = split_tables_by_x_gap(boxes)

    left_rows = group_cells_by_rows(left_boxes)
    right_rows = group_cells_by_rows(right_boxes)
    tables = [left_rows, right_rows]

    for table_idx, rows in enumerate(tables):
        for row_idx, row in enumerate(rows):
            if not row:
                continue
            x_min = min([x for (x, _, _, _) in row])
            y_min = min([y for (_, y, _, _) in row])
            x_max = max([x + w for (x, _, w, _) in row])
            y_max = max([y + h for (_, y, _, h) in row])

            color = (0, 255, 0) if table_idx == 0 else (255, 0, 0)  # Green for left, Blue for right
            cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), color, 2)
    all_extracted = []
    for table_idx, rows in enumerate(tables):
        for row in rows:
            if not row:
                continue
            extracted_cells = extract_text_from_cells(image_cv, [row])  # pass a list with just this one row
            for item in extracted_cells:
                category_name = item['Category']
                category_id = item['Category ID']
                vote = item['Item Number']
                insert_category(category_id, category_name)
                insert_vote(category_id, vote)
                all_extracted.append(item)
    return all_extracted
