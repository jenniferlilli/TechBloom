from flask import Flask, request, render_template, redirect, url_for, flash
from db_model import SessionLocal, Ballot, BallotVotes, OCRResult
from sqlalchemy.orm import joinedload
import boto3
import os

app = Flask(__name__, template_folder='.')
app.secret_key = 'secret-key'

s3 = boto3.client('s3')
BUCKET_NAME = 'techbloom-ballots'

@app.route('/review/<session_id>')
def review_dashboard(session_id):
    with SessionLocal() as db:
        error_ballots = db.query(Ballot).filter(
            Ballot.session_id == session_id,
            (Ballot.badge_id == None) | (Ballot.badge_id == '')
        ).all()
        unreadable_votes = db.query(BallotVotes).join(Ballot).filter(
            Ballot.session_id == session_id,
            BallotVotes.vote == 'unreadable'
        ).options(joinedload(BallotVotes.ballot)).all()
    return render_template('a_review_db.html', error_ballots=error_ballots, unreadable_votes=unreadable_votes)

@app.route('/fix_badge', methods=['POST'])
def fix_badge():
    ballot_id = request.form['ballot_id']
    new_badge = request.form['badge_id']
    with SessionLocal() as db:
        ballot = db.query(Ballot).get(ballot_id)
        if ballot:
            ballot.badge_id = new_badge
            db.commit()
    flash('Badge ID updated successfully')
    return redirect(request.referrer)

@app.route('/fix_vote', methods=['POST'])
def fix_vote():
    vote_id = request.form['vote_id']
    new_vote = request.form['vote']
    with SessionLocal() as db:
        vote = db.query(BallotVotes).get(vote_id)
        if vote:
            vote.vote = new_vote
            db.commit()
    flash('Vote updated successfully')
    return redirect(request.referrer)

@app.route('/delete_vote/<int:vote_id>')
def delete_vote(vote_id):
    with SessionLocal() as db:
        vote = db.query(BallotVotes).get(vote_id)
        if vote:
            db.delete(vote)
            db.commit()
    flash('Vote deleted successfully')
    return redirect(request.referrer)

@app.route('/delete_ballot/<int:ballot_id>')
def delete_ballot(ballot_id):
    with SessionLocal() as db:
        ballot = db.query(Ballot).get(ballot_id)
        if ballot:
            ocr_result = db.query(OCRResult).filter_by(session_id=ballot.session_id).first()
            if ocr_result:
                try:
                    s3.delete_object(Bucket=BUCKET_NAME, Key=ocr_result.filename)
                except Exception as e:
                    print("Error deleting from S3:", e)
            db.query(BallotVotes).filter_by(ballot_id=ballot.id).delete()
            db.delete(ballot)
            db.commit()
    flash('Ballot and associated data deleted')
    return redirect(request.referrer)

if __name__ == '__main__':
    app.run(debug=True)
