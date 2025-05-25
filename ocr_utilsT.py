from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/local/bin/tesseract'
import boto3
from io import BytesIO
from db_utils import get_connection, is_valid_badge, insert_ballot, update_ballot_status, insert_vote
import re


s3 = boto3.client('s3')


def parse_badge_id(text):
    for token in text.split():
        if token.isdigit():
            return token
    return None

def parse_votes(text):
    lines = text.splitlines()
    votes = {}
    i = 0

    while i < len(lines) - 2:
        category_line = lines[i].strip()
        code_line = lines[i + 1].strip()
        vote_line = lines[i + 2].strip()

        # Match a vote line with exactly three digits, possibly spaced like "0 | 0 | 1"
        if re.match(r'^\d\s*\|\s*\d\s*\|\s*\d$', vote_line):
            # Clean the vote digits (e.g., "0 | 0 | 1" → "001")
            vote_digits = ''.join(vote_line.split('|')).replace(' ', '')
            if vote_digits:
                votes[category_line] = vote_digits

        i += 1

    return votes
def process_local_ballot(image_path): #easier for immediate prining to dashboard
    with open(image_path, 'rb') as f:
        image = Image.open(f)
        text = pytesseract.image_to_string(image)

        badge_id = parse_badge_id(text)
        votes = parse_votes(text)
        return {
            "image": image_path,
            "badge_id": badge_id,
            "votes": votes
        }


def process_ballot_from_s3(s3_bucket, s3_key, session_id):

    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    img_bytes = obj['Body'].read()
    image = Image.open(BytesIO(img_bytes))

    text = pytesseract.image_to_string(image)

    badge_id = parse_badge_id(text)
    votes = parse_votes(text)

    ballot_id = insert_ballot(s3_key, session_id, badge_id=badge_id, status='pending')


    if badge_id and is_valid_badge(badge_id):

        for category, choice in votes.items():
            insert_vote(ballot_id, category, choice)
        update_ballot_status(ballot_id, 'processed')
        return {'status': 'processed', 'badge_id': badge_id, 'votes': votes}
    else:
        update_ballot_status(ballot_id, 'invalid')
        return {'status': 'invalid', 'badge_id': badge_id, 'votes': {}}
