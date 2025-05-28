# http://localhost:5000/login sigh
import os, uuid, boto3
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Text

from ocr_utilsT import extract_text_from_upload, extract_text_from_region
from io import BytesIO
import zipfile
from ocr_utilsT import extract_text_from_upload
from flask import jsonify
from botocore.exceptions import NoCredentialsError, ClientError

# sql setup
Base = declarative_base()
joinedSession = False


class BadgeID(Base):
    __tablename__ = 'badge_ids'
    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    badge_id = Column(String)


class UploadedZip(Base):
    __tablename__ = 'uploaded_zips'
    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    filename = Column(String)
    zip_data = Column(LargeBinary)


class UserSession(Base):
    __tablename__ = 'user_sessions'
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True)
    password = Column(String)


class OCRResult(Base):
    __tablename__ = 'ocr_results'
    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    filename = Column(String)
    extracted_text = Column(Text)


engine = create_engine('sqlite:///data.db')
Base.metadata.create_all(engine)
DBSession = sessionmaker(bind=engine)
db_session = DBSession()

s3 = boto3.client('s3')
bucket_name = 'techbloom-ballots'
app = Flask(__name__)
app.secret_key = 'secret-key'

ALLOWED_BADGE_EXTENSIONS = {'csv', 'txt'}
ALLOWED_ZIP_EXTENSIONS = {'zip'}



def upload_to_s3(file_obj, bucket, key):
    try:
        s3.upload_fileobj(file_obj, bucket, key)
        print(f"Uploaded to S3: {key}")
        return True
    except (NoCredentialsError, ClientError) as e:
        print(f"Upload failed: {e}")
        return False



def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions



def is_junk_file(file_info):
    filename = file_info.filename
    basename = os.path.basename(filename)

    return (
        filename.startswith('__MACOSX/') or           # macOS archive metadata
        '/__MACOSX/' in filename or                   # Catch nested __MACOSX
        basename.startswith('._') or                  # macOS resource forks
        basename.startswith('.') or                   # Hidden files (Linux/macOS)
        basename in ('Thumbs.db', 'desktop.ini') or   # Windows metadata files
        file_info.is_dir() or                         # Directories
        not basename.strip()                          # Empty or whitespace-only names
    )


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/create-session', methods=['GET', 'POST'])
def create_session():
    if request.method == 'POST':
        password = request.form.get('password')
        session_id = str(uuid.uuid4())[:8]
        db_session.add(UserSession(session_id=session_id, password=password))
        db_session.commit()
        session['session_id'] = session_id
        flash(f'Generated Session ID: {session_id}')
        return redirect(url_for('upload_files'))
    return render_template('createSession.html')


@app.route('/join-session', methods=['GET', 'POST'])
def join_session():
    if request.method == 'POST':
        session_id = request.form.get('session_id')
        password = request.form.get('password')
        user_session = db_session.query(UserSession).filter_by(session_id=session_id, password=password).first()
        if user_session:
            session['session_id'] = session_id
            session['joined_existing'] = True
            flash(f'Joined session: {session_id}')
            return redirect(url_for('upload_files'))
        else:
            flash('Invalid session ID or password.')
            return redirect(request.url)
    return render_template('joinSession.html')

@app.route('/upload-file', methods=['GET', 'POST'])
def upload_files():
    session_id = session.get('session_id')
    if not session_id:
        flash('Please log in or create a session first.')
        return redirect(url_for('login'))

    joined_existing = session.get('joined_existing', False)
    existing_badges = db_session.query(BadgeID).filter_by(session_id=session_id).first()
    existing_zip = db_session.query(UploadedZip).filter_by(session_id=session_id).first()

    # Handle raw image OCR API (likely JS frontend)
    if request.method == 'POST' and 'file' in request.files and len(request.files) == 1:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            image_data = file.read()
            try:
                text = extract_text_from_upload(image_data)
                return jsonify({'text': text})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

    # Handle full form POST with badge and zip
    if request.method == 'POST':
        badgeFile = request.files.get('badge_file')
        zipFile = request.files.get('zip_file')

        if not badgeFile and not zipFile and joined_existing:
            if joined_existing and (existing_badges or existing_zip):
                flash('Empty session, please reupload.')
                return redirect(url_for('dashboard'))
            else:
                flash('Please upload both badge and ZIP files.')
                return redirect(request.url)

        if badgeFile and allowed_file(badgeFile.filename, ALLOWED_BADGE_EXTENSIONS):
            badge_lines = badgeFile.read().decode('utf-8').splitlines()
            for line in badge_lines:
                badge_id = line.strip()
                if badge_id:
                    db_session.add(BadgeID(session_id=session_id, badge_id=badge_id))
            db_session.commit()
        elif badgeFile:
            flash('Invalid badge file. Must be .csv or .txt')
            return redirect(request.url)

        if zipFile and allowed_file(zipFile.filename, ALLOWED_ZIP_EXTENSIONS):
            filename = secure_filename(zipFile.filename)
            zip_bytes = zipFile.read()
            db_session.add(UploadedZip(session_id=session_id, filename=filename, zip_data=zip_bytes))
            db_session.commit()

            zip_stream = BytesIO(zip_bytes)
            with zipfile.ZipFile(zip_stream) as archive:
                for file_info in archive.infolist():
                    inner_filename = file_info.filename
                    if is_junk_file(file_info):
                        continue  # Skip junk files

                    if inner_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        with archive.open(file_info) as image_file:
                            key = f"{session_id}/{os.path.basename(inner_filename)}"
                            image_stream = BytesIO(image_file.read())
                            image_stream.seek(0)
                            if upload_to_s3(image_stream, bucket_name, key):
                                try:
                                    s3_object = s3.get_object(Bucket=bucket_name, Key=key)
                                    s3_bytes = s3_object['Body'].read()
                                    image_stream = BytesIO(s3_bytes)
                                    print(type(s3_bytes))
                                    print(len(s3_bytes))
                                    text = extract_text_from_upload(image_stream)
                                    print(f"OCR result for {inner_filename}:\n{text}\n")
                                    db_session.add(OCRResult(
                                        session_id=session_id,
                                        filename=inner_filename,
                                        extracted_text=text
                                    ))
                                except Exception as e:
                                    print(f"Failed to process {inner_filename}: {e}")
            db_session.commit()
        elif zipFile:
            flash('Invalid ZIP file.')
            return redirect(request.url)

        flash('Files uploaded successfully.')
        return redirect(url_for('dashboard'))

    return render_template('upload.html', joined_existing=joined_existing and (existing_badges or existing_zip))

@app.route('/dashboard')
def dashboard():
    session_id = session.get('session_id')
    if not session_id:
        flash('Please log in or create a session first.')
        return redirect(url_for('login'))

    badge_ids = db_session.query(BadgeID).filter_by(session_id=session_id).all()
    uploaded_zips = db_session.query(UploadedZip).filter_by(session_id=session_id).all()
    ocr_results = db_session.query(OCRResult).filter_by(session_id=session_id).all()
    return render_template('dashboard.html',
                           badge_ids=[b.badge_id for b in badge_ids],
                           zip_filenames=[z.filename for z in uploaded_zips],
                           ocr_results=ocr_results)

if __name__ == '__main__':
    app.run(debug=True)
