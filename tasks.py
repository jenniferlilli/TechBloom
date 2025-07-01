import zipfile
import os
from io import BytesIO
from celery import Celery
from easy_ocr import process_image 
from db_model import get_db_session, Ballot, OCRResult, BallotVotes
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

                    image_key = f"{session_id}/ballots/{file_info.filename}"
                    s3_client.upload_fileobj(BytesIO(image_data), bucket_name, image_key)

                    session_uuid = uuid.UUID(session_id)
                    new_ballot = Ballot(
                        session_id=session_uuid,
                        name=file_info.filename,
                        badge_status="readable",  
                        validity=True
                    )
                    db_session.add(new_ballot)
                    db_session.commit()

                    try:
                        extracted_votes = process_image(image_data, file_info.filename, session_id)
                        print("OCR extracted votes:", extracted_votes)

                        
                        db_session.add(OCRResult(
                            session_id=session_id,
                            filename=file_info.filename,
                            extracted_text=json.dumps(extracted_votes)
                        ))
                        db_session.commit()

                        for item in extracted_votes:
                            db_session.add(BallotVotes(
                                ballot_id=new_ballot.id,
                                category_id=item['category_id'],
                                vote=item['vote'],
                                vote_status=item['status'],
                                is_valid=item['status'] == 'readable',
                                key=item['key']
                            ))
                        db_session.commit()

                        processed_count += 1

                    except Exception as e:
                        print(f"OCR failed for {file_info.filename}: {e}")
                        
        return {'status': 'completed', 'processed_count': processed_count}

    except Exception as e:
        print(f"Error processing ZIP file: {e}")
        return {'status': 'failed', 'error': str(e)}

    finally:
        db_session.close()
        if os.path.exists(zip_path):
            os.remove(zip_path)
