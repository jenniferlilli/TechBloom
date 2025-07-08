import zipfile
import os
from io import BytesIO
from celery import Celery
from easy_ocr import process_image, readable_badge_id_exists, badge_id_exists
from db_model import get_db_session, Ballot, OCRResult, BallotVotes
from db_utils import insert_vote, insert_badge
import boto3
import json
from celery_app import make_celery
from PIL import Image
import numpy as np

celery = make_celery()

bucket_name = 'techbloom-ballots'
s3_client = boto3.client('s3')

@celery.task(bind=True)
def preprocess_zip_task(self, zip_key, session_id):
    db_session = get_db_session()
    processed_count = 0

    session_uuid = str(session_id)

    try:
       
        print(f"[Celery] Downloading ZIP from S3: {zip_key}")
        zip_obj = s3_client.get_object(Bucket=bucket_name, Key=zip_key)
        zip_bytes = zip_obj['Body'].read()
        zip_file = BytesIO(zip_bytes)

        with zipfile.ZipFile(zip_file, 'r') as archive:
            for file_info in archive.infolist():
                if not file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                print(f"[Celery] Processing image: {file_info.filename}")
                with archive.open(file_info) as image_file:
                    pil_image = Image.open(image_file).convert('RGB')
                    result = process_image(pil_image, file_info.filename)

                badge_id = result['badge_id']
                badge_key = result['badge_key']
                ocr_result = result['items']

                if badge_key == "" and (badge_id_exists(session_uuid, badge_id) and not readable_badge_id_exists(session_uuid, badge_id)):
                    validity = True
                    badge_status = 'readable'
                elif badge_key == "" and ((not badge_id_exists(session_uuid, badge_id)) or readable_badge_id_exists(session_uuid, badge_id)):
                    validity = False
                    badge_status = 'readable'
                else:
                    validity = True
                    badge_status = 'unreadable'

                ballot = Ballot(
                    session_id=session_uuid,
                    badge_id=badge_id,
                    badge_status=badge_status,
                    s3_key=badge_key,
                    name=file_info.filename,
                    validity=validity
                )
                db_session.add(ballot)
                db_session.flush()

                vote_objects = []
                for item in ocr_result:
                    vote_objects.append(BallotVotes(
                        badge_id=badge_id,
                        ballot_id=ballot.id,
                        name=file_info.filename,
                        category_id=item['Category ID'],
                        vote=item['Item Number'],
                        vote_status=item['Status'],
                        is_valid=validity,
                        key=item['Key']
                    ))
                db_session.add_all(vote_objects)

                db_session.add(OCRResult(
                    session_id=session_uuid,
                    filename=file_info.filename,
                    extracted_text=json.dumps(ocr_result)
                ))

                db_session.commit()
                processed_count += 1

        return {'status': 'completed', 'processed_count': processed_count}

    except Exception as e:
        db_session.rollback()
        print(f"[Celery] Error processing ZIP file: {e}")
        return {'status': 'failed', 'error': str(e)}

    finally:
        db_session.close()
