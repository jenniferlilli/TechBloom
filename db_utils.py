from sqlalchemy.orm import Session
from models import (
    UserSession, ValidBadgeIDs, UploadedZip, OCRResult,
    BallotVotes, BallotCategory, Ballot,
    SessionLocal
)

def insert_user_session(session_id: str, password: str):
    with SessionLocal() as session:
        user_session = UserSession(session_id=session_id, password=password)
        session.add(user_session)
        session.commit()

def validate_user_session(session_id: str, password: str):
    with SessionLocal() as session:
        return session.query(UserSession).filter_by(session_id=session_id, password=password).first()

def insert_valid_badge(session_id: str, badge_id: str):
    with SessionLocal() as session:
        badge = ValidBadgeIDs(session_id=session_id, badge_id=badge_id)
        session.add(badge)
        session.commit()

def insert_uploaded_zip(session_id: str, filename: str):
    with SessionLocal() as session:
        zip_file = UploadedZip(session_id=session_id, filename=filename)
        session.add(zip_file)
        session.commit()

def insert_ocr_result(session_id: str, filename: str, extracted_text: str):
    with SessionLocal() as session:
        ocr_result = OCRResult(session_id=session_id, filename=filename, extracted_text=extracted_text)
        session.add(ocr_result)
        session.commit()

def insert_vote(category_id: str, vote: str):
    with SessionLocal() as session:
        vote_obj = BallotVotes(category_id=category_id, vote=vote)
        session.add(vote_obj)
        session.commit()

def insert_category(category_id: str, category_name: str):
    with SessionLocal() as session:
        category = BallotCategory(category_id=category_id, category_name=category_name)
        session.add(category)
        session.commit()

def insert_badge(session_id: str, badge_id: str):
    with SessionLocal() as session:
        ballot = Ballot(session_id=session_id, badge_id=badge_id)
        session.add(ballot)
        session.commit()
