from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/local/bin/tesseract'
from io import BytesIO

def extract_text_from_upload(file_input):
    if isinstance(file_input, bytes):
        # If it's raw bytes (from a zip), wrap in BytesIO
        image = Image.open(file_input)
    else:
        # If it's a FileStorage (from direct upload), use its stream
        image = Image.open(file_input)

    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_region(file_storage):

    image = Image.open(file_storage)
    # Example: Crop if needed -> image = image.crop((left, top, right, bottom))
    text = pytesseract.image_to_string(image)
    return text
