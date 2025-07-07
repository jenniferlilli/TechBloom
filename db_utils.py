import uuid

from sqlalchemy.orm import Session
from db_model import (
    UserSession, ValidBadgeIDs, UploadedZip, OCRResult,
    BallotVotes, Ballot, Product,
    SessionLocal
)

def insert_user_session(session_id: str, password: str):
    with SessionLocal() as db_session:
        session_id = uuid.UUID(session_id)
        new_session = UserSession(session_id=session_id, password=password)
        db_session.add(new_session)
        db_session.commit()
        db_session.close()

def validate_user_session(session_id: str, password: str):
    with SessionLocal() as session:
        session_id = uuid.UUID(session_id)
        return session.query(UserSession).filter_by(session_id=session_id, password=password).first()

def insert_valid_badge(session_id: str, badge_id: str):
    with SessionLocal() as session:
        session_id = uuid.UUID(session_id)
        badge = ValidBadgeIDs(session_id=session_id, badge_id=badge_id)
        session.add(badge)
        session.commit()

def insert_uploaded_zip(session_id: str, filename: str):
    with SessionLocal() as session:
        session_id = uuid.UUID(session_id)
        zip_file = UploadedZip(session_id=session_id, filename=filename)
        session.add(zip_file)
        session.commit()

def insert_ocr_result(session_id: str, filename: str, extracted_text: str):
    with SessionLocal() as session:
        session_id = uuid.UUID(session_id)
        ocr_result = OCRResult(session_id=session_id, filename=filename, extracted_text=extracted_text)
        session.add(ocr_result)
        session.commit()


def insert_vote(badge_id: str, file_name: str, category_id: str, vote: str, status: str, validity: bool, key: str, session_id: str):
    with SessionLocal() as session:
        session_id = uuid.UUID(session_id)
        ballot = session.query(Ballot).filter(
            Ballot.name == file_name,
            Ballot.session_id == session_id
        ).first()

        if ballot:
            ballot_id = ballot.id
        else:
            ballot_id = None

        vote_obj = BallotVotes(
            badge_id=badge_id,
            ballot_id=ballot_id,
            name=file_name,
            category_id=category_id,
            vote=vote,
            vote_status=status,
            is_valid=validity,
            key=key
        )
        session.add(vote_obj)
        session.commit()

def insert_badge(session_id: str, badge_id: str, status: str, key: str, name: str, validity: bool):
    with SessionLocal() as session:
        session_id = uuid.UUID(session_id)

        ballot = Ballot(session_id=session_id, badge_id=badge_id, badge_status=status, s3_key=key, name=name,validity=validity)
        session.add(ballot)
        session.commit()

def insert_products(session_id: str, products: list[tuple[str, str, str]]):
    with SessionLocal() as session:
        session.query(Product).filter_by(session_id=session_id).delete()

        for category_id, product_number, product_name in products:
            product = Product(
                session_id=session_id,
                category_id=category_id,
                product_number=product_number,
                product_name=product_name
            )
            session.add(product)

        session.commit()
