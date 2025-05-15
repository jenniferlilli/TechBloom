#http://localhost:5000/login sigh
import os, uuid
import zipfile
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

session_store = {}

app = Flask(__name__)
app.secret_key = 'secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

ALLOWED_BADGE_EXTENSIONS = {'csv', 'txt'}
ALLOWED_ZIP_EXTENSIONS = {'zip'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        action = request.form.get('action')
        password = request.form.get('password')

        if action == 'create':
            session_id = str(uuid.uuid4())[:8]
            session_store[session_id] = password
            session['session_id'] = session_id
            flash(f'Created session: {session_id}')
            return redirect(url_for('upload_files'))

        elif action == 'join':
            session_id = request.form.get('session_id')
            if session_id in session_store and session_store[session_id] == password:
                session['session_id'] = session_id
                flash(f'Joined session: {session_id}')
                return redirect(url_for('upload_files'))
            else:
                flash('Invalid session ID or password.')
                return redirect(request.url)

    return render_template('login.html')

def extractIDS(filepath):
    badge_ids = set()
    with open(filepath, 'r') as f:
        for line in f:
            for item in line.strip().split(','):
                badge_ids.add(item.strip())
    return list(badge_ids)

def extractImages(zipPath, extractTo):
    import zipfile
    imagePaths = []
    with zipfile.ZipFile(zipPath, 'r') as zipRef:
        zipRef.extractall(extractTo)
        for root, _, files in os.walk(extractTo):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    imagePaths.append(os.path.join(root, file))
    return imagePaths

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        badgeFile = request.files.get('badge_file')
        zipFile = request.files.get('zip_file')

        if badgeFile and allowed_file(badgeFile.filename, ALLOWED_BADGE_EXTENSIONS):
            badgeFileName = secure_filename(badgeFile.filename)
            badgePath = os.path.join(app.config['UPLOAD_FOLDER'], badgeFileName)
            badgeFile.save(badgePath)
        else:
            flash('Invalid badge file. Must be .csv or .txt')
            return redirect(request.url)

        if zipFile and allowed_file(zipFile.filename, ALLOWED_ZIP_EXTENSIONS):
            zipFileName = secure_filename(zipFile.filename)
            zipPath = os.path.join(app.config['UPLOAD_FOLDER'], zipFileName)
            zipFile.save(zipPath)
        else:
            flash('Invalid ZIP file.')
            return redirect(request.url)

        badgeIds = extractIDS(badgePath)
        
        imageFolder = os.path.join(app.config['UPLOAD_FOLDER'], 'session_' + session['session_id'])
        os.makedirs(imageFolder, exist_ok=True)
        imagePaths = extractImages(zipPath, imageFolder)

        session['badgeIds'] = badgeIds
        session['imagePaths'] = imagePaths

        flash('Files uploaded successfully!')
        return redirect(url_for('upload_files'))

    return render_template('upload.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
