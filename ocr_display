import re

VALID_CATEGORIES = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
    "KA", "KB", "KC", "L", "M", "N", "O", "P", "PA", "Q", "QA",
    "FA", "FB", "FC", "FD", "FE", "FF", "FG", "FH", "RA", "S", "T",
    "U", "V", "W", "WA", "X", "Y", "YA",
}

def extract_reg_id(text):
    match = re.search(r"REG ID#?\s*(\d{5})", text)
    if match:
        return match.group(1)
    return None

def clean_number(raw_number):
    # Remove unwanted characters
    for ch in [' ', ',', '/', '\\', '|', '"', "'"]:
        raw_number = raw_number.replace(ch, '')

    cleaned = raw_number.strip("()")

    if not cleaned:
        return ""

    if len(cleaned) <= 2 or len(cleaned) > 7:
        return None

    # Step 1: Gradually strip '1's until down to <= 3 digits
    while len(cleaned) > 3 and '1' in cleaned:
        cleaned = cleaned.replace('1', '', 1)

    # Step 2: Further reduce if needed
    while len(cleaned) > 3:
        doubled = False
        for i in range(len(cleaned) - 1):
            if cleaned[i] == cleaned[i + 1]:
                cleaned = cleaned[:i] + cleaned[i + 1:]
                doubled = True
                break
        if doubled:
            continue
        cleaned = cleaned[:-1]

    if len(cleaned) == 3 and cleaned.isdigit():
        return cleaned
    return None


def is_number_line(line):
    return bool(re.match(r"^[\d\s,()/\\|\"']+$", line.strip()))

def find_categories_and_numbers(lines):
    results = {}
    line_count = len(lines)
   
    lines = [l.strip() for l in lines]
    i = 0
    while i < line_count:
        line = lines[i]
       
        # Check exact category line
        if line in VALID_CATEGORIES:
            category = line
            number_text = ""
           
            if i + 1 < line_count:
                next_line = lines[i+1]
                if is_number_line(next_line):
                    number_text = next_line
                    i += 1
                else:
                    if i + 2 < line_count and is_number_line(lines[i+2]):
                        number_text = lines[i+2]
                        i += 2
           
            results[category] = number_text
       
        else:
            # NEW: Check if line starts with a category + number attached
            for cat in VALID_CATEGORIES:
                if line.startswith(cat):
                    rest = line[len(cat):].strip()
                    if rest and all(c in "0123456789(), " for c in rest):
                        # assign category cat with number rest
                        results[cat] = rest
                        break
       
        # Special case for 0 and ( as before
        if line == "0":
            if i == 0 or lines[i-1] not in VALID_CATEGORIES:
                results["O"] = ""
                if i + 1 < line_count and is_number_line(lines[i+1]):
                    results["O"] = lines[i+1]
                    i += 1
        elif line == "(":
            if i == 0 or lines[i-1] not in VALID_CATEGORIES:
                results["C"] = ""
                if i + 1 < line_count and is_number_line(lines[i+1]):
                    results["C"] = lines[i+1]
                    i += 1
        elif line.startswith("(") and len(line) > 1:
            if "C" not in results:
                results["C"] = line
       
        i += 1
   
    return results

def postprocess_results(results):
    final_results = {}
    for cat in VALID_CATEGORIES:
        raw_num = results.get(cat, None)
        if raw_num is None:
            final_results[cat] = f"Category {cat} not found"
            continue
       
        cleaned = clean_number(raw_num)
        if cleaned is None:
            final_results[cat] = f"Category {cat} has a bad number"
            continue
        if cleaned == "":
            final_results[cat] = f"Category {cat} not found"
            continue
       
        final_results[cat] = cleaned
    return final_results

def ocr_cleaning_algorithm(text):
    reg_id = extract_reg_id(text)
    lines = text.splitlines()
    cat_numbers_raw = find_categories_and_numbers(lines)
    final = postprocess_results(cat_numbers_raw)
   
    output = f"REG ID#: {reg_id}\n\n"
    for cat in sorted(VALID_CATEGORIES):
        output += f"{cat}: {final[cat]}\n"
    return output


with open('sample5.txt', 'r', encoding='utf-8') as file:
    input_text = file.read()
print(ocr_cleaning_algorithm(input_text))
