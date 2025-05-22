from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import zipfile
from io import BytesIO
import boto3
from ocr_utils import extract_text_from_upload, extract_text_from_region

routes = Blueprint('routes', __name__)


S3_BUCKET = os.getenv('S3_BUCKET', 'techbloom-ballots')  # <- set S3 bucket name
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_ZIP_EXTENSIONS = {'zip'}

s3 = boto3.client('s3')

def allowed_file(filename, allowed_exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

@routes.route('/upload/image', methods=['POST'])
def upload_single_image():
    file = request.files.get('image_file')
    if not file or not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'error': 'Invalid or missing image file.'}), 400

    filename = secure_filename(file.filename)
    session_id = request.cookies.get('SESSION_ID', 'default')
    s3_key = f"{session_id}/{filename}"

    # Upload file directly to S3
    # Environment must have AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY set
    s3.upload_fileobj(file.stream, S3_BUCKET, s3_key)

    file.stream.seek(0)
    text = extract_text_from_upload(file)

    return jsonify({'text': text, 's3_key': s3_key})

@routes.route('/upload/zip', methods=['POST'])
def upload_zip_images():
    zip_file = request.files.get('zip_file')
    if not zip_file or not allowed_file(zip_file.filename, ALLOWED_ZIP_EXTENSIONS):
        return jsonify({'error': 'Invalid or missing zip file.'}), 400

    zip_bytes = zip_file.read()
    zip_stream = BytesIO(zip_bytes)
    try:
        zf = zipfile.ZipFile(zip_stream)
    except zipfile.BadZipFile:
        return jsonify({'Error': 'Uploaded file is not a valid ZIP archive.'}), 400

    results = []
    session_id = request.cookies.get('SESSION_ID', 'default')

    for info in zf.infolist():
        name = info.filename
        ext = name.rsplit('.', 1)[-1].lower()
        if ext in ALLOWED_IMAGE_EXTENSIONS:
            try:
                with zf.open(info) as image_file:
                    file_bytes = image_file.read()
                    image_stream = BytesIO(file_bytes)
                    safe_name = secure_filename(name)
                    s3_key = f"{session_id}/{safe_name}"

                    s3.upload_fileobj(BytesIO(file_bytes), S3_BUCKET, s3_key)

                    text = extract_text_from_region(file_storage=image_stream) if False else extract_text_from_upload(file_storage=BytesIO(file_bytes))
                    results.append({'file': name, 'text': text, 's3_key': s3_key})
            except Exception as e:
                results.append({'file': name, 'error': str(e)})
        else:
            results.append({'file': name, 'error': 'Unsupported extension'})

    return jsonify({'results': results})
