import io
import os
import uuid

import boto3
import easyocr #not used rn, will use, have time for a backup
import cv2
import numpy as np
import timm
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from PIL import Image
from db_utils import insert_vote, insert_badge
from db_model import get_db_session, ValidBadgeIDs, Ballot

model = timm.create_model("resnet18", pretrained=False, num_classes=10)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://huggingface.co/gpcarl123/resnet18_mnist/resolve/main/resnet18_mnist.pth",
        map_location="cpu",
        file_name="resnet18_mnist.pth",
    )
)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def extract_and_normalize_largest_digit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    height, width = binary.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.9 * width), 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    binary = cv2.subtract(binary, detected_lines)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(binary, kernel_dilate, iterations=1)

    height, width = dilated.shape
    dilated[:int(0.1 * height), :] = 0
    dilated[-int(0.1 * height):, :] = 0
    dilated[:, :int(0.1 * width)] = 0
    dilated[:, -int(0.1 * width):] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    image_center = np.array([width / 2, height / 2])
    best_label = -1
    best_score = -np.inf

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 80 or h > 0.9 * height:
            continue
        cx, cy = centroids[label]
        dist2 = (cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2
        density = area / (w * h + 1e-5)
        score = area * density - 0.1 * dist2
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == -1:
        print(f"[!] No valid digit found.")
        return None

    selected = {best_label}
    queue = [best_label]
    margin = 50

    while queue:
        label = queue.pop()
        x, y, w, h, _ = stats[label]
        grow_x1 = max(x - margin, 0)
        grow_y1 = max(y - margin, 0)
        grow_x2 = min(x + w + margin, width)
        grow_y2 = min(y + h + margin, height)

        for other_label in range(1, num_labels):
            if other_label in selected:
                continue
            ox, oy, ow, oh, oa = stats[other_label]
            if oa < 50 or oh > 0.9 * height or ow > 0.9 * width:
                continue
            if ox + ow < grow_x1 or ox > grow_x2 or oy + oh < grow_y1 or oy > grow_y2:
                continue
            selected.add(other_label)
            queue.append(other_label)

    selected_centroids = [centroids[i] for i in selected]
    for other_label in range(1, num_labels):
        if other_label in selected:
            continue
        if stats[other_label][4] < 50:
            continue
        dist = torch.cdist(
            torch.tensor(np.array([centroids[other_label]]), dtype=torch.float32),
            torch.tensor(np.array(selected_centroids), dtype=torch.float32)
        )
        if dist.min().item() < 80:
            selected.add(other_label)

    merged_mask = np.zeros_like(dilated, dtype=np.uint8)
    for label in selected:
        merged_mask[labels == label] = 255

    ys, xs = np.where(merged_mask)
    if len(xs) == 0 or len(ys) == 0:
        print("[!] Empty merged digit.")
        return None
    x1, x2 = np.min(xs), np.max(xs)
    y1, y2 = np.min(ys), np.max(ys)

    pad = 10
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, width)
    y2 = min(y2 + pad, height)

    digit_crop = merged_mask[y1:y2 + 1, x1:x2 + 1]
    h_new, w_new = digit_crop.shape
    scale = 20.0 / max(h_new, w_new)
    resized_digit = cv2.resize(digit_crop, (int(w_new * scale), int(h_new * scale)), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    rh, rw = resized_digit.shape
    x_offset = (28 - rw) // 2
    y_offset = (28 - rh) // 2
    canvas[y_offset:y_offset + rh, x_offset:x_offset + rw] = resized_digit

    digit_resized = canvas

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

    edges = cv2.Canny(binary, 30, 150, apertureSize=3, L2gradient=True)

    lines = []

    hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    if hough_lines is not None:
        lines.extend(hough_lines[:, 0].tolist())

    houghp_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                   minLineLength=min(image.shape[1] // 4, 50),
                                   maxLineGap=10)
    if houghp_lines is not None:
        for x1, y1, x2, y2 in houghp_lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            angle = np.arctan2(dy, dx)
            lines.append([0, angle])

    if not lines:
        return image

    angles = []
    for line in lines:
        if len(line) == 2:
            rho, theta = line
            angle = np.degrees(theta) - 90
        else:
            angle = np.degrees(line[1]) - 90

        if -10 < angle < 10:
            angles.append(angle)

    if not angles:
        return image

    mean_angle = np.mean(angles)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, mean_angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    return deskewed

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
    return binary

def split_roi_into_digit_boxes(image, expected_rows=5):

    # Step 1: Enhanced preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    # Step 2: Detect REG box
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reg_rect = None
    max_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > max_area and 2 < (w / h) < 10:
            max_area = area
            reg_rect = (x, y, w, h)

    if reg_rect is None:
        print("[!] Could not find REG box.")
        return []

    x, y, w, h = reg_rect
    roi = image[y:y + h, x:x + w]

    # Step 3: Clean horizontal underline detection
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_roi = clahe.apply(gray_roi)

    th = cv2.adaptiveThreshold(
        enhanced_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    morph = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    underline_boxes = []
    for cnt in contours:
        ux, uy, uw, uh = cv2.boundingRect(cnt)
        aspect = uw / uh if uh > 0 else 0
        if 8 <= uw <= roi.shape[1] and 1 <= uh <= 6 and aspect > 10:
            underline_boxes.append((ux, uy, uw, uh))

    if len(underline_boxes) < expected_rows:
        print(f"[!] Only found {len(underline_boxes)} underlines.")
        return []

    underline_boxes = sorted(underline_boxes, key=lambda b: b[1])[:expected_rows]
    left_x = min(b[0] for b in underline_boxes)
    left_x = min(b[0] for b in underline_boxes)
    right_x = max(b[0] + b[2] for b in underline_boxes)

    digit_boxes = []
    output = roi.copy()

    for i, (ux, uy, uw, uh) in enumerate(underline_boxes[:1]):
        line_area = th[:uy, left_x:right_x]

        long_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        long_line = cv2.morphologyEx(line_area, cv2.MORPH_OPEN, long_kernel, iterations=1)
        top_contours, _ = cv2.findContours(long_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        top_y = 0
        if top_contours:
            top_y = max([cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] for cnt in top_contours])

        cropped_row = roi[top_y:uy, left_x:right_x]
        row_gray = cv2.cvtColor(cropped_row, cv2.COLOR_BGR2GRAY)

        _, row_binary = cv2.threshold(row_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(row_binary, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

        digit_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_contours = sorted(digit_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

        combined_contours = []
        if digit_contours:
            combined_contours = [digit_contours[0]]

            for cnt in digit_contours[1:]:
                x1, y1, w1, h1 = cv2.boundingRect(combined_contours[-1])
                x2, y2, w2, h2 = cv2.boundingRect(cnt)

                if x2 <= (x1 + w1 + 5):
                    combined_x = min(x1, x2)
                    combined_y = min(y1, y2)
                    combined_w = max(x1 + w1, x2 + w2) - combined_x
                    combined_h = max(y1 + h1, y2 + h2) - combined_y

                    combined_contours[-1] = np.array([[
                        [combined_x, combined_y],
                        [combined_x + combined_w, combined_y],
                        [combined_x + combined_w, combined_y + combined_h],
                        [combined_x, combined_y + combined_h]
                    ]])
                else:
                    combined_contours.append(cnt)

        for j, dc in enumerate(combined_contours):
            dx, dy, dw, dh = cv2.boundingRect(dc)
            area = dw * dh
            aspect = dh / dw if dw > 0 else 0

            if dh > 10 and dw > 5 and area > 50:
                padding = 5
                pad_top = max(dy - padding, 0)
                pad_bottom = min(dy + dh + padding, cropped_row.shape[0])
                pad_left = max(dx - padding, 0)
                pad_right = min(dx + dw + padding, cropped_row.shape[1])

                digit = cropped_row[pad_top:pad_bottom, pad_left:pad_right]
                digit_gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)

                digit_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
                digit_enhanced = digit_clahe.apply(digit_gray)

                _, digit_binary = cv2.threshold(digit_enhanced, 0, 255,
                                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                digit_clean = cv2.erode(digit_binary,
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                                        iterations=1)

                digit_boxes.append(digit_clean)

    print(f"[✓] Extracted {len(digit_boxes)} digits from REG box.")
    return digit_boxes, roi


def upload_badge_to_s3(image, file_name, object_prefix="low_confidence_badge"):
    s3 = boto3.client("s3")
    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode image to JPEG format.")
    buffer = io.BytesIO(encoded_image.tobytes())

    base_name = os.path.splitext(file_name)[0]
    key = f"{object_prefix}/{base_name}.jpg"

    s3.upload_fileobj(
        Fileobj=buffer,
        Bucket='techbloom-ballots',
        Key=key,
        ExtraArgs={"ContentType": "image/jpeg"}
    )

    print(f"[S3] Uploaded to s3://techbloom-ballots/{key}")
    return key

def upload_vote_to_s3(image, file_name, object_prefix="low_confidence_vote"):
    s3 = boto3.client("s3")
    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode image to JPEG format.")
    buffer = io.BytesIO(encoded_image.tobytes())

    base_name = os.path.splitext(file_name)[0]

    # Generate a unique UUID string
    unique_id = str(uuid.uuid4())

    # Construct the key with UUID appended
    key = f"{object_prefix}/{base_name}_{unique_id}.jpg"

    s3.upload_fileobj(
        Fileobj=buffer,
        Bucket='techbloom-ballots',
        Key=key,
        ExtraArgs={"ContentType": "image/jpeg"}
    )

    print(f"[S3] Uploaded to s3://techbloom-ballots/{key}")
    return key

def process_badge_id(image, model, file_name):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    h, w = image.shape[:2]
    roi = image[0:int(h * 0.25), int(w * 0.65):w]
    digit_boxes, extracted_img = split_roi_into_digit_boxes(roi)

    digit_string = ""
    low_confidence = False
    for i, digit_img in enumerate(digit_boxes):
        if len(digit_img.shape) == 3:
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

        _, digit_binary = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        orig_h, orig_w = digit_binary.shape
        aspect_ratio = orig_w / float(orig_h)

        centered = center_digit_proportional(digit_binary)

        if orig_h > orig_w:
            new_h = 24
            new_w = max(4, int(new_h * aspect_ratio))
        else:
            new_w = 24
            new_h = int(new_w / aspect_ratio)

        resized = cv2.resize(centered, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        processed = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        processed[start_y:start_y + new_h, start_x:start_x + new_w] = resized

        if aspect_ratio < 0.5:
            kernel = np.ones((3, 1), np.uint8)
            processed = cv2.dilate(processed, kernel, iterations=1)

        input_tensor = transform(processed).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
            pred = np.argmax(probs)
            confidence = probs[pred]
        if confidence < 0.7:
            digit_string += "?"
            low_confidence = True
        else:
            digit_string += str(pred)
    if low_confidence or not len(digit_string) == 5:
        key = upload_badge_to_s3(extracted_img, file_name)
        return digit_string, key
    else:
        return digit_string, ""

def center_digit_proportional(img):
    pts = cv2.findNonZero(img)
    if pts is None:
        return img

    x, y, w, h = cv2.boundingRect(pts)
    digit_only = img[y:y + h, x:x + w]

    padded = np.zeros((h + 4, w + 4), dtype=np.uint8)
    padded[2:2 + h, 2:2 + w] = digit_only

    return padded

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
        x, y, w, h = cv2.boundingRect(cnt)
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
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
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

def extract_digits(cell_img, file_name):
    h, w = cell_img.shape[:2]
    segment_width = w // 3
    digits = []
    good_vote = True
    for i in range(3):
        start_x = i * segment_width
        end_x = (i + 1) * segment_width if i < 2 else w
        digit_img = cell_img[:, start_x:end_x]

        norm_digit = extract_and_normalize_largest_digit(digit_img)
        print(f"Segment {i} norm_digit is None: {norm_digit is None}")
        if norm_digit is not None:

            if isinstance(norm_digit, np.ndarray):
                norm_digit = transforms.ToPILImage()(norm_digit)

            input_tensor = transform(norm_digit).unsqueeze(0)

            device = next(model.parameters()).device
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)[0].cpu().numpy()

            pred_class = int(np.argmax(probabilities))
            confidence = probabilities[pred_class]

            print(f"Segment {i} predicted digit: {pred_class}")
            print(f"Segment {i} confidences: " + ", ".join(f"{d}:{p:.2f}" for d, p in enumerate(probabilities)))
            if confidence < 0.70:
                digits.append('?')
                good_vote = False
            else:
                digits.append(str(pred_class))
        else:
            print(f"No digit extracted for segment {i}")
            digits.append('?')
    final = ''.join(digits)
    print(f"Full 3-digit result: {final}")
    key = ""
    if not len(final) == 3:
        good_vote = False
    if not good_vote:
        key = upload_vote_to_s3(cell_img, file_name)
    return final, key
    
def extract_text_from_cells(image, rows, count, file_name):
    extracted = []
    CATEGORY_IDS = [
    "A", "B", "C", "D", "E", "G", "H", "I", "J", "F",
    "FA", "FB", "FC", "FD", "FE", "FF", "FG", "FH",
    "K", "KA", "KB", "KC", "L", "M", "N", "O", "P", "PA",
    "Q", "QA", "R", "RA", "S", "T", "U", "V", "W",
    "WA", "X", "Y", "YA"
    ]
    for row in rows:
        row = sorted(row, key=lambda b: b[0])
        cells = []
        key = ""
        for i, (x, y, w, h) in enumerate(row):
            cell_img = image[y:y + h, x:x + w]
            if i == 2:
                processed = preprocess_cell(cell_img)
                item_number, key = extract_digits(processed, file_name)
                cells.append(item_number)

        cat_id = CATEGORY_IDS[count]
        item_no = cells[0] if len(cells) > 0 else ''
        count += 1

        if len(item_no) == 3 and item_no.find("?") == -1: #TO EDIT FOR REVIEW W/ IF ELSE
            print(f"[DEBUG] Processing row {count}, length of row: {len(row)}")
            extracted.append({
                'Category ID': cat_id,
                'Item Number': item_no,
                'Status' : 'readable',
                'Key' : key
            })
        else:
            print(f"Not valid vote {item_no}.")
            extracted.append({
                'Category ID': cat_id,
                'Item Number' : item_no,
                'Status' : 'unreadable',
                'Key' : key
            })
    return extracted

def badge_id_exists(session_id: str, badge_id: str) -> bool:
    session = get_db_session()
    try:
        exists = session.query(ValidBadgeIDs).filter(
            ValidBadgeIDs.session_id == session_id,
            ValidBadgeIDs.badge_id == badge_id
        ).first() is not None
    finally:
        session.close()
    return exists

def readable_badge_id_exists(session_id: str, badge_id: str) -> bool:
    session = get_db_session()
    try:
        exists = session.query(Ballot).filter(
            Ballot.session_id == session_id,
            Ballot.badge_id == badge_id,
            Ballot.badge_status == 'readable'
        ).first() is not None
    finally:
        session.close()
    return exists

def process_image(image_bytes, file_name, session_id: str):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    image_cv = deskew_image(image_cv)
    badge_id, key = process_badge_id(image_cv, model, file_name)
    validity = True

    if key == "" and (badge_id_exists(session_id, badge_id) and not readable_badge_id_exists(session_id, badge_id)):
        insert_badge(session_id, badge_id, 'readable', key, validity)
    elif key == "" and ((not badge_id_exists(session_id, badge_id)) or readable_badge_id_exists(session_id, badge_id)):
        validity = False
        insert_badge(session_id, badge_id, 'readable', key, validity)
    else:
        insert_badge(session_id, badge_id, 'unreadable', key, validity)

    print(f"Extracted Badge ID: {badge_id}")

    boxes = detect_table_cells(image_cv)
    boxes = filter_valid_boxes(boxes, min_y=450)
    print(f"Found {len(boxes)} boxes")

    left_boxes, right_boxes = split_tables_by_x_gap(boxes)

    left_rows = group_cells_by_rows(left_boxes)
    right_rows = group_cells_by_rows(right_boxes)
    tables = [left_rows, right_rows]

    all_extracted = []
    count = 0
    for table_idx, rows in enumerate(tables):
        if table_idx == 0:
            rows = rows[1:]
        extracted_cells = extract_text_from_cells(image_cv, rows, count, file_name)
        for item in extracted_cells:
            category_id = item['Category ID']
            vote = item['Item Number']
            status = item['Status']
            key = item ['Key']
            insert_vote(badge_id, file_name, category_id, vote, status, validity, key)
            all_extracted.append(item)
        count += len(rows)
    return all_extracted