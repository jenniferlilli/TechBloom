# models.py
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

Base = declarative_base()

class ValidBadgeIDs(Base):
    __tablename__ = 'valid_badge_ids'
    session_id = Column(String, nullable=False, primary_key=True)
    badge_id = Column(String, ForeignKey('sessions.id'), nullable=False, primary_key=True)

class Ballot(Base):
    __tablename__ = 'ballots'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False)
    badge_id = Column(String, nullable=False)
    name = Column(String)
    created_at = Column(DateTime, server_default=func.current_timestamp())

class UploadedZip(Base):
    __tablename__ = 'uploaded_zips'
    session_id = Column(String, primary_key=True, unique=True)
    filename = Column(String)

class UserSession(Base):
    __tablename__ = 'sessions'
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True)
    password = Column(String)

class OCRResult(Base):
    __tablename__ = 'ocr_results'
    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    filename = Column(String)
    extracted_text = Column(String)  # for debugging

class BallotCategory(Base):
    __tablename__ = 'categories'
    category_id = Column(String, primary_key=True, nullable=False)
    category_name = Column(String, nullable=False)

class BallotVotes(Base):
    __tablename__ = 'votes'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ballot_id = Column(Integer, ForeignKey('ballots.id'), nullable=False)
    category_id = Column(String, ForeignKey('categories.category_id'), nullable=False)
    vote = Column(String)

engine = create_engine('sqlite:///data.db')
Base.metadata.create_all(engine)

SessionLocal = scoped_session(sessionmaker(bind=engine))