# models.py
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, func
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
    name = Column(String)
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

class BallotCategory(Base):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key = True, autoincrement = True)
    category_id = Column(String)
    category_name = Column(String)

class BallotVotes(Base):
    __tablename__ = 'votes'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ballot_id = Column(Integer, ForeignKey('ballots.id'))
    category_id = Column(String, ForeignKey('categories.category_id'))
    vote = Column(String)

engine = create_engine('sqlite:///data.db')
Base.metadata.create_all(engine)

SessionLocal = scoped_session(sessionmaker(bind=engine))