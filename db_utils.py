from sqlalchemy.orm import Session
from db_model import (
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

def insert_vote(category_id: str, vote: str, status: str):
    with SessionLocal() as session:
        vote_obj = BallotVotes(category_id=category_id, vote=vote, vote_status=status)
        session.add(vote_obj)
        session.commit()

def insert_category(category_id: str):
    with SessionLocal() as session:
        category = BallotCategory(category_id=category_id)
        session.add(category)
        session.commit()

def insert_badge(session_id: str, badge_id: str, status: str):
    with SessionLocal() as session:
        ballot = Ballot(session_id=session_id, badge_id=badge_id, badge_status=status)
        session.add(ballot)
        session.commit()


def get_ocr_results_by_session(session_id: str):
    
    '''
    with SessionLocal() as session:
        return session.query(OCRResult).filter_by(session_id=session_id).all()
    '''
    return [
    {
        "Filename": "ballot1.jpg",
        "Extracted Text": [
            {"Category ID": "A", "Item Number": "101"},
            {"Category ID": "B", "Item Number": "102"},
            {"Category ID": "C", "Item Number": "103"},
            {"Category ID": "D", "Item Number": "104"},
            {"Category ID": "E", "Item Number": "105"},
            {"Category ID": "G", "Item Number": "106"},
            {"Category ID": "H", "Item Number": "107"},
            {"Category ID": "I", "Item Number": "108"},
            {"Category ID": "J", "Item Number": "109"},
            {"Category ID": "F", "Item Number": "110"},
            {"Category ID": "FA", "Item Number": "111"},
            {"Category ID": "FB", "Item Number": "112"},
            {"Category ID": "FC", "Item Number": "113"},
            {"Category ID": "FD", "Item Number": "114"},
            {"Category ID": "FE", "Item Number": "115"},
            {"Category ID": "FF", "Item Number": "116"},
            {"Category ID": "FG", "Item Number": "117"},
            {"Category ID": "FH", "Item Number": "118"},
            {"Category ID": "K", "Item Number": "119"},
            {"Category ID": "KB", "Item Number": "120"},
            {"Category ID": "KC", "Item Number": "121"},
            {"Category ID": "L", "Item Number": "122"},
            {"Category ID": "M", "Item Number": "123"},
            {"Category ID": "N", "Item Number": "124"},
            {"Category ID": "O", "Item Number": "125"},
            {"Category ID": "P", "Item Number": "126"},
            {"Category ID": "PA", "Item Number": "127"},
            {"Category ID": "Q", "Item Number": "128"},
            {"Category ID": "QA", "Item Number": "129"},
            {"Category ID": "R", "Item Number": "130"},
            {"Category ID": "RA", "Item Number": "131"},
            {"Category ID": "S", "Item Number": "132"},
            {"Category ID": "T", "Item Number": "133"},
            {"Category ID": "U", "Item Number": "134"},
            {"Category ID": "V", "Item Number": "135"},
            {"Category ID": "W", "Item Number": "136"},
            {"Category ID": "WA", "Item Number": "137"},
            {"Category ID": "X", "Item Number": "138"},
            {"Category ID": "Y", "Item Number": "139"},
            {"Category ID": "YA", "Item Number": "140"}
        ]
    },
    {
        "Filename": "ballot2.jpg",
        "Extracted Text": [
            {"Category ID": "A", "Item Number": "201"},
            {"Category ID": "B", "Item Number": "202"},
            {"Category ID": "C", "Item Number": "203"},
            {"Category ID": "D", "Item Number": "204"},
            {"Category ID": "E", "Item Number": "205"},
            {"Category ID": "G", "Item Number": "206"},
            {"Category ID": "H", "Item Number": "207"},
            {"Category ID": "I", "Item Number": "208"},
            {"Category ID": "J", "Item Number": "209"},
            {"Category ID": "F", "Item Number": "210"},
            {"Category ID": "FA", "Item Number": "211"},
            {"Category ID": "FB", "Item Number": "212"},
            {"Category ID": "FC", "Item Number": "213"},
            {"Category ID": "FD", "Item Number": "214"},
            {"Category ID": "FE", "Item Number": "215"},
            {"Category ID": "FF", "Item Number": "216"},
            {"Category ID": "FG", "Item Number": "217"},
            {"Category ID": "FH", "Item Number": "218"},
            {"Category ID": "K", "Item Number": "219"},
            {"Category ID": "KB", "Item Number": "220"},
            {"Category ID": "KC", "Item Number": "221"},
            {"Category ID": "L", "Item Number": "222"},
            {"Category ID": "M", "Item Number": "223"},
            {"Category ID": "N", "Item Number": "224"},
            {"Category ID": "O", "Item Number": "225"},
            {"Category ID": "P", "Item Number": "226"},
            {"Category ID": "PA", "Item Number": "227"},
            {"Category ID": "Q", "Item Number": "228"},
            {"Category ID": "QA", "Item Number": "229"},
            {"Category ID": "R", "Item Number": "230"},
            {"Category ID": "RA", "Item Number": "231"},
            {"Category ID": "S", "Item Number": "232"},
            {"Category ID": "T", "Item Number": "233"},
            {"Category ID": "U", "Item Number": "234"},
            {"Category ID": "V", "Item Number": "235"},
            {"Category ID": "W", "Item Number": "236"},
            {"Category ID": "WA", "Item Number": "237"},
            {"Category ID": "X", "Item Number": "238"},
            {"Category ID": "Y", "Item Number": "239"},
            {"Category ID": "YA", "Item Number": "240"}
        ]
    },
    {
        "Filename": "ballot3.jpg",
        "Extracted Text": [
            {"Category ID": "A", "Item Number": "101"},
            {"Category ID": "B", "Item Number": "102"},
            {"Category ID": "C", "Item Number": "103"},
            {"Category ID": "D", "Item Number": "204"},
            {"Category ID": "E", "Item Number": "205"},
            {"Category ID": "G", "Item Number": "106"},
            {"Category ID": "H", "Item Number": "107"},
            {"Category ID": "I", "Item Number": "208"},
            {"Category ID": "J", "Item Number": "209"},
            {"Category ID": "F", "Item Number": "110"},
            {"Category ID": "FA", "Item Number": "211"},
            {"Category ID": "FB", "Item Number": "112"},
            {"Category ID": "FC", "Item Number": "213"},
            {"Category ID": "FD", "Item Number": "114"},
            {"Category ID": "FE", "Item Number": "115"},
            {"Category ID": "FF", "Item Number": "216"},
            {"Category ID": "FG", "Item Number": "117"},
            {"Category ID": "FH", "Item Number": "218"},
            {"Category ID": "K", "Item Number": "119"},
            {"Category ID": "KB", "Item Number": "220"},
            {"Category ID": "KC", "Item Number": "121"},
            {"Category ID": "L", "Item Number": "222"},
            {"Category ID": "M", "Item Number": "123"},
            {"Category ID": "N", "Item Number": "124"},
            {"Category ID": "O", "Item Number": "225"},
            {"Category ID": "P", "Item Number": "126"},
            {"Category ID": "PA", "Item Number": "127"},
            {"Category ID": "Q", "Item Number": "228"},
            {"Category ID": "QA", "Item Number": "129"},
            {"Category ID": "R", "Item Number": "130"},
            {"Category ID": "RA", "Item Number": "231"},
            {"Category ID": "S", "Item Number": "232"},
            {"Category ID": "T", "Item Number": "133"},
            {"Category ID": "U", "Item Number": "134"},
            {"Category ID": "V", "Item Number": "135"},
            {"Category ID": "W", "Item Number": "236"},
            {"Category ID": "WA", "Item Number": "137"},
            {"Category ID": "X", "Item Number": "138"},
            {"Category ID": "Y", "Item Number": "139"},
            {"Category ID": "YA", "Item Number": "140"}
        ]
    },
    {
        "Filename": "ballot4.jpg",
        "Extracted Text": [
            {"Category ID": "A", "Item Number": "201"},
            {"Category ID": "B", "Item Number": "202"},
            {"Category ID": "C", "Item Number": "203"},
            {"Category ID": "D", "Item Number": "104"},
            {"Category ID": "E", "Item Number": "105"},
            {"Category ID": "G", "Item Number": "206"},
            {"Category ID": "H", "Item Number": "207"},
            {"Category ID": "I", "Item Number": "108"},
            {"Category ID": "J", "Item Number": "109"},
            {"Category ID": "F", "Item Number": "210"},
            {"Category ID": "FA", "Item Number": "111"},
            {"Category ID": "FB", "Item Number": "212"},
            {"Category ID": "FC", "Item Number": "113"},
            {"Category ID": "FD", "Item Number": "214"},
            {"Category ID": "FE", "Item Number": "215"},
            {"Category ID": "FF", "Item Number": "116"},
            {"Category ID": "FG", "Item Number": "217"},
            {"Category ID": "FH", "Item Number": "118"},
            {"Category ID": "K", "Item Number": "219"},
            {"Category ID": "KB", "Item Number": "120"},
            {"Category ID": "KC", "Item Number": "221"},
            {"Category ID": "L", "Item Number": "122"},
            {"Category ID": "M", "Item Number": "223"},
            {"Category ID": "N", "Item Number": "224"},
            {"Category ID": "O", "Item Number": "125"},
            {"Category ID": "P", "Item Number": "226"},
            {"Category ID": "PA", "Item Number": "227"},
            {"Category ID": "Q", "Item Number": "128"},
            {"Category ID": "QA", "Item Number": "229"},
            {"Category ID": "R", "Item Number": "230"},
            {"Category ID": "RA", "Item Number": "131"},
            {"Category ID": "S", "Item Number": "132"},
            {"Category ID": "T", "Item Number": "233"},
            {"Category ID": "U", "Item Number": "234"},
            {"Category ID": "V", "Item Number": "235"},
            {"Category ID": "W", "Item Number": "136"},
            {"Category ID": "WA", "Item Number": "237"},
            {"Category ID": "X", "Item Number": "238"},
            {"Category ID": "Y", "Item Number": "239"},
            {"Category ID": "YA", "Item Number": "240"}
        ]
    },
    {
        "Filename": "ballot5.jpg",
        "Extracted Text": [
            {"Category ID": "A", "Item Number": "301"},
            {"Category ID": "B", "Item Number": "302"},
            {"Category ID": "C", "Item Number": "303"},
            {"Category ID": "D", "Item Number": "304"},
            {"Category ID": "E", "Item Number": "305"},
            {"Category ID": "G", "Item Number": "306"},
            {"Category ID": "H", "Item Number": "307"},
            {"Category ID": "I", "Item Number": "308"},
            {"Category ID": "J", "Item Number": "309"},
            {"Category ID": "F", "Item Number": "310"},
            {"Category ID": "FA", "Item Number": "311"},
            {"Category ID": "FB", "Item Number": "312"},
            {"Category ID": "FC", "Item Number": "313"},
            {"Category ID": "FD", "Item Number": "314"},
            {"Category ID": "FE", "Item Number": "315"},
            {"Category ID": "FF", "Item Number": "316"},
            {"Category ID": "FG", "Item Number": "317"},
            {"Category ID": "FH", "Item Number": "318"},
            {"Category ID": "K", "Item Number": "319"},
            {"Category ID": "KB", "Item Number": "320"},
            {"Category ID": "KC", "Item Number": "321"},
            {"Category ID": "L", "Item Number": "322"},
            {"Category ID": "M", "Item Number": "323"},
            {"Category ID": "N", "Item Number": "324"},
            {"Category ID": "O", "Item Number": "325"},
            {"Category ID": "P", "Item Number": "326"},
            {"Category ID": "PA", "Item Number": "327"},
            {"Category ID": "Q", "Item Number": "328"},
            {"Category ID": "QA", "Item Number": "329"},
            {"Category ID": "R", "Item Number": "330"},
            {"Category ID": "RA", "Item Number": "331"},
            {"Category ID": "S", "Item Number": "332"},
            {"Category ID": "T", "Item Number": "333"},
            {"Category ID": "U", "Item Number": "334"},
            {"Category ID": "V", "Item Number": "335"},
            {"Category ID": "W", "Item Number": "336"},
            {"Category ID": "WA", "Item Number": "337"},
            {"Category ID": "X", "Item Number": "338"},
            {"Category ID": "Y", "Item Number": "339"},
            {"Category ID": "YA", "Item Number": "340"}
        ]
    }
]