from PIL import Image
import pytesseract

def extract_text_from_upload(file_storage):

    image = Image.open(file_storage)
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_region(file_storage):

    image = Image.open(file_storage)
    # Example: Crop if needed -> image = image.crop((left, top, right, bottom))
    text = pytesseract.image_to_string(image)
    return text
