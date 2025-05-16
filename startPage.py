#http://localhost:5000/login sigh
import os, uuid
import zipfile
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

session_store = {}

app = Flask(__name__)
app.secret_key = 'secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max????

ALLOWED_BADGE_EXTENSIONS = {'csv', 'txt'}
ALLOWED_ZIP_EXTENSIONS = {'zip'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/create-session', methods=['GET', 'POST'])
def create_session():
    if request.method == 'POST':
        password = request.form.get('password')
        session_id = str(uuid.uuid4())[:8]
        session_store[session_id] = password
        session['session_id'] = session_id
        flash(f'Generated Session ID: {session_id}')
        return redirect(url_for('upload_files'))

    return render_template('createSession.html')

@app.route('/join-session', methods=['GET', 'POST'])
def join_session():
    if request.method == 'POST':
        session_id = request.form.get('session_id')
        password = request.form.get('password')
        if session_id in session_store and session_store[session_id] == password:
            session['session_id'] = session_id
            flash(f'Joined session: {session_id}')
            return redirect(url_for('upload_files'))
        else:
            flash('Invalid session ID or password.')
            return redirect(request.url)
    return render_template('joinSession.html')


@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        badgeFile = request.files.get('badge_file')
        zipFile = request.files.get('zip_file')

        if badgeFile and allowed_file(badgeFile.filename, ALLOWED_BADGE_EXTENSIONS):
            badgePath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(badgeFile.filename))
            badgeFile.save(badgePath)
        else:
            flash('Invalid badge file. Must be .csv or .txt')
            return redirect(request.url)

        if zipFile and allowed_file(zipFile.filename, ALLOWED_ZIP_EXTENSIONS):
            zipPath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(zipFile.filename))
            zipFile.save(zipPath)
        else:
            flash('Invalid ZIP file.')
            return redirect(request.url)

        flash('Files uploaded successfully! You can now process them.')
        return redirect(url_for('dashboard'))

    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('dashboard.html', uploaded_files=uploaded_files)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
