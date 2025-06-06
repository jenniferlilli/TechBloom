# models.py
from sqlalchemy import create_engine, Column, Integer, Boolean, String, ForeignKey, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

Base = declarative_base()

class ValidBadgeIDs(Base):
    __tablename__ = 'valid_badge_ids'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    badge_id = Column(String)

class Ballot(Base):
    __tablename__ = 'ballots'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    badge_id = Column(String)
    badge_status = Column(String)
    s3_key = Column(String)
    name = Column(String)
    validity = Column(Boolean)
    created_at = Column(DateTime, server_default=func.current_timestamp())

class UploadedZip(Base):
    __tablename__ = 'uploaded_zips'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String)
    filename = Column(String)

class UserSession(Base):
    __tablename__ = 'sessions'
    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    password = Column(String)

class OCRResult(Base):
    __tablename__ = 'ocr_results'
    id = Column(Integer, primary_key=True, autoincrement = True)
    session_id = Column(String, ForeignKey('sessions.id'))
    filename = Column(String)
    extracted_text = Column(String)  # for debugging

class BallotVotes(Base):
    __tablename__ = 'votes'
    id = Column(Integer, primary_key=True, autoincrement=True)
    badge_id = Column(String)
    ballot_id = Column(Integer, ForeignKey('ballots.id'))
    name = Column(String)
    category_id = Column(String)
    vote = Column(String)
    key = Column(String)
    vote_status = Column(String)
    is_valid = Column(Boolean)

engine = create_engine('sqlite:///data.db')
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(engine)

SessionLocal = scoped_session(sessionmaker(bind=engine))

def get_db_session():
    return SessionLocal()
