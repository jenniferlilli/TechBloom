import os
import uuid
import boto3
import json
import re
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from sqlalchemy import func, desc
from werkzeug.utils import secure_filename
from db_model import (
    ValidBadgeIDs,
    Ballot,
    UploadedZip,
    UserSession,
    OCRResult,
    BallotVotes,
    SessionLocal,
)
from easy_ocr import process_image, badge_id_exists, readable_badge_id_exists
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

@app.route('/logout')
def logout():
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
            try:
                badge_lines = badgeFile.read().decode('utf-8').splitlines()
            except UnicodeDecodeError as e:
                flash('Badge file must be UTF-8 encoded text (.csv or .txt).', 'error')
                db_session.close()
                return redirect(request.url)
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
                                existing_file = db_session.query(OCRResult).filter_by(filename=file_info.filename,
                                                                                      session_id=session_id).first()
                                if existing_file:
                                    print(f"Skipping duplicate file: {file_info.filename}")
                                    continue

                                with archive.open(file_info) as image_file:
                                    image_data = image_file.read()
                                    image_key = f"{session_id}/ballots/{file_info.filename}"
                                    upload_to_s3(BytesIO(image_data), bucket_name, image_key)
                                    try:
                                        ocr_text = process_image(image_data, file_info.filename, session_id)
                                        db_session.add(OCRResult(
                                            session_id=session_id,
                                            filename=file_info.filename,
                                            extracted_text=json.dumps(ocr_text)
                                        ))
                                        db_session.commit()
                                        processed_count += 1
                                    except Exception as e:
                                        print(f"OCR failed for {file_info.filename}: {e}")
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

    db_session = get_db_session()

    vote_records = (
        db_session.query(BallotVotes)
        .join(Ballot, BallotVotes.badge_id == Ballot.badge_id)
        .filter(
            Ballot.session_id == session_id,
            Ballot.badge_status == 'readable',
            Ballot.validity == True,
            BallotVotes.is_valid == True,
            BallotVotes.vote_status == 'readable'
        )
        .all()
    )

    category_votes = defaultdict(list)

    for vote in vote_records:
        cleaned_vote = (vote.vote or "").strip()
        if vote.category_id:
            category_votes[vote.category_id.upper()].append(cleaned_vote)

    top3_per_category = {}

    for category, votes in category_votes.items():
        counts = Counter(votes)
        top_votes = counts.most_common(3)  # top 3 votes, no unreadable category needed

        top3_per_category[category] = top_votes
    print(top3_per_category)
    db_session.close()
    return render_template("a_dashboard.html", top3_per_category=top3_per_category)

@app.route('/review')
def review_dashboard():
    db_session = get_db_session()

    ballots_with_badge_issues = (
        db_session.query(Ballot)
        .filter(Ballot.badge_status == 'unreadable')
        .all()
    )
    badges_data = []
    for ballot in ballots_with_badge_issues:
        s3_url = None
        if ballot.s3_key:
            s3_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': ballot.s3_key},
                ExpiresIn=3600
            )
        badges_data.append({
            'name': ballot.name,
            'ballot_id' : ballot.id,
            'badge_id': ballot.badge_id,
            's3_url': s3_url,
        })

    votes_with_errors = (
        db_session.query(BallotVotes)
        .join(Ballot, BallotVotes.badge_id == Ballot.badge_id)
        .filter(
            Ballot.badge_status == 'readable',
            Ballot.validity == True,
            BallotVotes.vote_status == 'unreadable',
            BallotVotes.is_valid == True
        )
        .all()
    )
    votes_data = []
    for vote in votes_with_errors:
        s3_url = None
        if vote.key:
            s3_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': vote.key},
                ExpiresIn=3600
            )
        votes_data.append({
            'vote_id': vote.id,
            'category': vote.category_id,
            'current_vote': vote.vote,
            'badge_id': vote.badge_id,
            's3_url': s3_url,
            'name': vote.name
        })
    print(votes_data)
    db_session.close()
    return render_template('a_review_db.html', badges=badges_data, votes=votes_data)

@app.route('/fix_vote', methods=['POST'])
def fix_vote():
    session_id = session.get('session_id')
    vote_id = request.form.get('vote_id')
    new_vote = request.form.get('vote', '').strip()

    if not vote_id or not new_vote:
        flash('Invalid input. Please provide a vote.', 'error')
        return redirect(request.referrer or url_for('review_dashboard'))

    db_session = get_db_session()
    vote = db_session.query(BallotVotes).get(vote_id)

    if vote is None:
        flash('Vote not found.', 'error')
        db_session.close()
        return redirect(request.referrer or url_for('review_dashboard'))

    vote.vote = new_vote
    vote.vote_status = 'readable'

    if vote.key:
        try:
            s3.delete_object(Bucket=bucket_name, Key=vote.key)
            vote.key = ""  # Clear the key after deletion
        except Exception as e:
            print(f"Failed to delete S3 object {vote.key}: {e}")

    db_session.commit()

    flash(f'Vote updated successfully for badge {vote.badge_id}.', 'success')
    db_session.close()
    return redirect(request.referrer or url_for('review_dashboard'))

@app.route('/fix_badge', methods=['POST'])
def fix_badge():
    session_id = session.get('session_id')
    ballot_id = request.form['ballot_id']
    new_badge = request.form['badge_id'].strip()

    db_session = get_db_session()
    ballot = db_session.query(Ballot).get(ballot_id)
    if not ballot:
        flash('Ballot not found.')
        db_session.close()
        return redirect(request.referrer)

    old_s3_key = ballot.s3_key

    is_valid = badge_id_exists(session_id, new_badge)
    is_duplicate = readable_badge_id_exists(session_id, new_badge)

    if is_duplicate:
        flash('Badge ID already exists.')
        db_session.close()
        return redirect(request.referrer)

    if not is_valid:
        flash('Badge ID does not exist.')
        db_session.close()
        return redirect(request.referrer)

    ballot.badge_id = new_badge
    ballot.badge_status = 'readable'
    ballot.validity = is_valid
    ballot.s3_key = ""
    db_session.commit()

    votes = db_session.query(BallotVotes).filter(BallotVotes.ballot_id == ballot_id).all()
    for vote in votes:
        vote.badge_id = new_badge
        vote.validity = is_valid
    db_session.commit()

    if old_s3_key:
        try:
            s3.delete_object(Bucket=bucket_name, Key=old_s3_key)
        except Exception as e:
            print(f"Failed to delete S3 object {old_s3_key}: {e}")

    db_session.close()
    flash('Badge ID updated successfully and validity checked.')
    return redirect(url_for('review_dashboard'))

@app.route('/delete_vote/<int:vote_id>')
def delete_vote(vote_id):
    db_session = get_db_session()
    vote = db_session.query(BallotVotes).get(vote_id)

    if vote:
        if vote.key:
            try:
                s3.delete_object(Bucket=bucket_name, Key=vote.key)
            except Exception as e:
                print(f"Error deleting vote image {vote.key} from S3:", e)

        db_session.delete(vote)
        db_session.commit()
        flash('Vote deleted successfully', 'success')
    else:
        flash('Vote not found', 'error')

    db_session.close()
    return redirect(request.referrer or url_for('review_dashboard'))

@app.route('/delete_ballot/<int:ballot_id>')
def delete_ballot(ballot_id):
    db_session = get_db_session()
    ballot = db_session.query(Ballot).get(ballot_id)

    if ballot:
        ballot_id = ballot.id
        badge_id = ballot.badge_id
        session_id = ballot.session_id

        # Delete ballot image from S3
        if ballot.s3_key:
            try:
                s3.delete_object(Bucket=bucket_name, Key=ballot.s3_key)
            except Exception as e:
                print(f"Error deleting ballot S3 image: {e}")

        # Delete OCRResult S3 image and DB row
        ocr_result = db_session.query(OCRResult).filter_by(session_id=session_id, filename=ballot.name).first()
        if ocr_result:
            if ocr_result.filename:
                try:
                    s3.delete_object(Bucket=bucket_name, Key=ocr_result.filename)
                except Exception as e:
                    print(f"Error deleting OCR S3 image: {e}")
            db_session.delete(ocr_result)

        # Delete all BallotVotes with matching badge_id
        votes = db_session.query(BallotVotes).filter_by(ballot_id=ballot_id).all()
        for vote in votes:
            if vote.key:
                try:
                    s3.delete_object(Bucket=bucket_name, Key=vote.key)
                except Exception as e:
                    print(f"Error deleting vote image {vote.key} from S3: {e}")
            db_session.delete(vote)

        # Delete all Ballots with same badge_id
        db_session.delete(ballot)

        db_session.commit()
        flash(f'Deleted badge ID "{badge_id}", all associated ballots, votes, and OCR result.', 'success')
    else:
        flash('Ballot not found.', 'error')

    db_session.close()
    return redirect(request.referrer or url_for('review_dashboard'))

if __name__ == '__main__':
    print("Starting Flask app…")
    app.run(debug=True)
