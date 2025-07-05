import zipfile
import os
from io import BytesIO
from celery import Celery
from easy_ocr import process_image, readable_badge_id_exists, badge_id_exists
from db_model import get_db_session, Ballot, OCRResult, BallotVotes
from db_utils import insert_vote, insert_badge
import boto3
import json
import uuid
from celery_app import make_celery

celery = make_celery()

bucket_name = 'techbloom-ballots'
s3_client = boto3.client('s3')


@celery.task(bind=True)
def preprocess_zip_task(self, zip_path, session_id):
    db_session = get_db_session()
    processed_count = 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as archive:
            for file_info in archive.infolist():
                if file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"Processing image: {file_info.filename}")

                    with archive.open(file_info) as image_file:
                        image_data = image_file.read()

                    result = process_image(image_data, file_info.filename)
                    badge_id = result['badge_id']
                    badge_key = result['badge_key']
                    ocr_result = result['items']
                    validity = True
                    if badge_key == "" and (badge_id_exists(session_id, badge_id) and not readable_badge_id_exists(session_id, badge_id)):
                        insert_badge(session_id, badge_id, 'readable', badge_key, file_info.filename, validity)
                    elif badge_key == "" and ((not badge_id_exists(session_id, badge_id)) or readable_badge_id_exists(session_id, badge_id)):
                        validity = False
                        insert_badge(session_id, badge_id, 'readable', badge_key, file_info.filename, validity)
                    else:
                        insert_badge(session_id, badge_id, 'unreadable', badge_key, file_info.filename, validity)

                    for item in ocr_result:
                        category_id = item['Category ID']
                        vote = item['Item Number']
                        status = item['Status']
                        key = item['Key']
                        insert_vote(badge_id, file_info.filename, category_id, vote, status, validity, key, session_id)

                    db_session.add(OCRResult(
                        session_id=session_id,
                        filename=file_info.filename,
                        extracted_text=json.dumps(ocr_result)
                    ))


                    db_session.commit()

                    processed_count += 1

        return {'status': 'completed', 'processed_count': processed_count}

    except Exception as e:
        print(f"Error processing ZIP file: {e}")
        return {'status': 'failed', 'error': str(e)}

    finally:
        db_session.close()
        if os.path.exists(zip_path):
            os.remove(zip_path)
