from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import zipfile
from io import BytesIO
import boto3
from ocr_utilsT import process_ballot_from_s3
from db_utils import init_db

routes = Blueprint('routes', __name__)


init_db()

S3_BUCKET = os.getenv('S3_BUCKET', 'techbloom-ballots')
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

    s3.upload_fileobj(file.stream, S3_BUCKET, s3_key)

    # Process ballot
    result = process_ballot_from_s3(S3_BUCKET, s3_key, session_id)
    return jsonify(result)

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
                s3_key = f"{session_id}/{secure_filename(name)}"
                # upload to S3
                s3.upload_fileobj(zf.open(info), S3_BUCKET, s3_key)
                # process each image
                result = process_ballot_from_s3(S3_BUCKET, s3_key, session_id)
                results.append({**result, 'file': name})
            except Exception as e:
                results.append({'file': name, 'error': str(e)})
        else:
            results.append({'file': name, 'error': 'Unsupported extension'})

    return jsonify({'results': results})
