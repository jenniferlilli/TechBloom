import os
import uuid
import boto3
import json
import re
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from db_utils import get_ocr_results_by_session
from db_model import (
    ValidBadgeIDs,
    Ballot,
    UploadedZip,
    UserSession,
    OCRResult,
    BallotCategory,
    BallotVotes,
    SessionLocal,
    get_all_ballots_with_missing_badge,
    get_all_unreadable_votes
)
from easy_ocr import process_image
from io import BytesIO
import zipfile
from botocore.exceptions import NoCredentialsError, ClientError
from collections import defaultdict, Counter

s3 = boto3.client('s3')
bucket_name = 'techbloom-ballots'

app = Flask(__name__, template_folder='.')
app.secret_key = 'secret-key' 

ALLOWED_BADGE_EXTENSIONS = {'csv', 'txt'}
ALLOWED_ZIP_EXTENSIONS = {'zip'}

def get_db_session():
    return SessionLocal()

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
        filename.startswith('__MACOSX/') or
        '/__MACOSX/' in filename or
        basename.startswith('._') or
        basename.startswith('.') or
        basename in ('Thumbs.db', 'desktop.ini') or
        file_info.is_dir() or
        not basename.strip()
    )

@app.route('/login')
def login():
    return render_template('a_login.html')

@app.route('/create-session', methods=['GET', 'POST'])
def create_session():
    if request.method == 'POST':
        password = request.form.get('password')
        session_id = str(uuid.uuid4())[:8]
        db_session = get_db_session()
        db_session.add(UserSession(session_id=session_id, password=password))
        db_session.commit()
        db_session.close()
        session['session_id'] = session_id
        flash(f'Generated Session ID: {session_id}')
        return redirect(url_for('upload_files'))
    return render_template('a_createSession.html')


@app.route('/join-session', methods=['GET', 'POST'])
def join_session():
    if request.method == 'POST':
        session_id = request.form.get('session_id')
        password = request.form.get('password')
        db_session = get_db_session()
        user_session = db_session.query(UserSession).filter_by(
            session_id=session_id, password=password
        ).first()
        db_session.close()
        if user_session:
            session['session_id'] = session_id
            session['joined_existing'] = True
            flash(f'Joined session: {session_id}')
            return redirect(url_for('upload_files'))
        else:
            flash('Invalid session ID or password.')
            return redirect(request.url)
    return render_template('a_joinSession.html')


@app.route('/upload-file', methods=['GET', 'POST'])
def upload_files():
    session_id = session.get('session_id')
    if not session_id:
        flash('Please log in or create a session first.')
        return redirect(url_for('login'))

    db_session = get_db_session()
    joined_existing = session.get('joined_existing', False)
    existing_badges = db_session.query(ValidBadgeIDs).filter_by(session_id=session_id).first()
    existing_zip = db_session.query(UploadedZip).filter_by(session_id=session_id).first()

    if request.method == 'POST':
        badgeFile = request.files.get('badge_file')
        zipFile = request.files.get('zip_file')

        if not badgeFile and not zipFile and joined_existing:
            if joined_existing and (existing_badges or existing_zip):
                flash('Empty session, please reupload.')
                db_session.close()
                return redirect(url_for('dashboard'))
            else:
                flash('Please upload both badge and ZIP files.')
                db_session.close()
                return redirect(request.url)

        if badgeFile and allowed_file(badgeFile.filename, ALLOWED_BADGE_EXTENSIONS):
            badge_lines = badgeFile.read().decode('utf-8').splitlines()
            for line in badge_lines:
                badge_id = line.strip()
                if badge_id:
                    db_session.add(ValidBadgeIDs(session_id=session_id, badge_id=badge_id))
            db_session.commit()
            flash('Badge IDs uploaded successfully.')
        elif badgeFile:
            flash('Invalid badge file. Must be .csv or .txt')
            db_session.close()
            return redirect(request.url)

 
        if zipFile and allowed_file(zipFile.filename, ALLOWED_ZIP_EXTENSIONS):
            filename = secure_filename(zipFile.filename)
            zip_bytes = zipFile.read()
            zip_key = f'{session_id}/{filename}'

            if upload_to_s3(BytesIO(zip_bytes), bucket_name, zip_key):
                db_session.add(UploadedZip(session_id=session_id, filename=filename))
                db_session.commit()

                zip_stream = BytesIO(zip_bytes)
                try:
                    with zipfile.ZipFile(zip_stream) as archive:
                        processed_count = 0
                        for file_info in archive.infolist():
                            if is_junk_file(file_info):
                                continue
                            if file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                with archive.open(file_info) as image_file:
                                    image_data = image_file.read()
                                    try:
                                        ocr_text = process_image(image_data)
                                        db_session.add(OCRResult(
                                            session_id=session_id,
                                            filename=file_info.filename,
                                            extracted_text=json.dumps(ocr_text)
                                        ))
                                        processed_count += 1
                                    except Exception as e:
                                        print(f"OCR failed for {file_info.filename}: {e}")
                        db_session.commit()
                        flash(f'Successfully processed {processed_count} ballot images.')
                except zipfile.BadZipFile:
                    flash('Invalid ZIP file.')
                    db_session.close()
                    return redirect(request.url)
            else:
                flash('Failed to upload ZIP to S3.')
                db_session.close()
                return redirect(request.url)
        elif zipFile:
            flash('Invalid file type. ZIP required.')
            db_session.close()
            return redirect(request.url)

        db_session.close()
        return redirect(url_for('dashboard'))

    db_session.close()
    return render_template('a_upload.html', joined_existing=joined_existing and (existing_badges or existing_zip))


@app.route('/dashboard')
def dashboard():
    session_id = session.get("session_id")
    if not session_id:
        flash('Please log in or create a session first.')
        return redirect(url_for('login'))

    ocr_results = get_ocr_results_by_session(session_id)
    item_pattern = re.compile(r"^\d{3}$")

    category_votes = defaultdict(list)
    for result in ocr_results:
        extracted_data = result.get("Extracted Text", [])
        for entry in extracted_data:
            category_id = entry.get("Category ID", "").strip().upper()
            item_number = entry.get("Item Number", "").strip()
            if category_id and item_pattern.match(item_number):
                category_votes[category_id].append(item_number)

    top3_per_category = {}
    for category, items in category_votes.items():
        counts = Counter(items)
        top3 = counts.most_common(3)
        top3_per_category[category] = top3

    return render_template("a_dashboard.html", top3_per_category=top3_per_category)

@app.route('/review')
def review_dashboard():
    session = SessionLocal()

    error_ballots = session.query(Ballot).filter((Ballot.badge_id == None) | (Ballot.badge_id == '')).all()
    unreadable_votes = session.query(BallotVotes).filter(BallotVotes.vote == 'unreadable').all()

    session.close()
    return render_template('a_review_db.html', error_ballots=error_ballots, unreadable_votes=unreadable_votes)


@app.route('/fix_badge', methods=['POST'])
def fix_badge():
    ballot_id = request.form['ballot_id']
    new_badge = request.form['badge_id']
    db_session = get_db_session()
    ballot = db_session.query(Ballot).get(ballot_id)
    if ballot:
        ballot.badge_id = new_badge
        db_session.commit()
    db_session.close()
    flash('Badge ID updated successfully')
    return redirect(request.referrer)

@app.route('/fix_vote', methods=['POST'])
def fix_vote():
    vote_id = request.form['vote_id']
    new_vote = request.form['vote']
    db_session = get_db_session()
    vote = db_session.query(BallotVotes).get(vote_id)
    if vote:
        vote.vote = new_vote
        db_session.commit()
    db_session.close()
    flash('Vote updated successfully')
    return redirect(request.referrer)

@app.route('/delete_vote/<int:vote_id>')
def delete_vote(vote_id):
    db_session = get_db_session()
    vote = db_session.query(BallotVotes).get(vote_id)
    if vote:
        db_session.delete(vote)
        db_session.commit()
    db_session.close()
    flash('Vote deleted successfully')
    return redirect(request.referrer)

@app.route('/delete_ballot/<int:ballot_id>')
def delete_ballot(ballot_id):
    db_session = get_db_session()
    ballot = db_session.query(Ballot).get(ballot_id)
    if ballot:
        ocr_result = db_session.query(OCRResult).filter_by(session_id=ballot.session_id).first()
        if ocr_result:
            try:
                s3.delete_object(Bucket=bucket_name, Key=ocr_result.filename)
            except Exception as e:
                print("Error deleting from S3:", e)

        db_session.query(BallotVotes).filter_by(ballot_id=ballot.id).delete()
        db_session.delete(ballot)
        db_session.commit()
    db_session.close()
    flash('Ballot and associated data deleted')
    return redirect(request.referrer)

@app.route('/seed_review_test')
def seed_review_test():
    session = SessionLocal()

    bad_ballot = Ballot(session_id="test_session", badge_id='', name="Test Ballot")
    session.add(bad_ballot)
    session.flush()

    unreadable_vote = BallotVotes(ballot_id=bad_ballot.id, category_id='A', vote='unreadable')
    session.add(unreadable_vote)

    session.commit()
    session.close()

    return "Seeded dummy review data"

if __name__ == '__main__':
    print("Starting Flask app…")
    app.run(debug=True)
