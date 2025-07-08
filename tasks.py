import zipfile
import os
import json
from io import BytesIO
from celery_app import make_celery
from easy_ocr import process_image, readable_badge_id_exists, badge_id_exists
from db_model import get_db_session, Ballot, OCRResult, BallotVotes
import boto3

celery = make_celery()

bucket_name = 'techbloom-ballots'
s3_client = boto3.client('s3')

@celery.task(bind=True)
def preprocess_zip_task(self, zip_s3_key, session_id):
    db_session = get_db_session()
    processed_count = 0
    session_uuid = str(session_id)

    tmp_zip_path = f"/tmp/{os.path.basename(zip_s3_key)}"

    try:
        print(f"[Celery] Downloading ZIP from S3: {zip_s3_key}")
        # Download ZIP file from S3 to /tmp
        s3_client.download_file(bucket_name, zip_s3_key, tmp_zip_path)

        with zipfile.ZipFile(tmp_zip_path, 'r') as archive:
            for file_info in archive.infolist():
                if not file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                print(f"[Celery] Processing image: {file_info.filename}")
                with archive.open(file_info) as image_file:
                    image_data = image_file.read()

                result = process_image(image_data, file_info.filename)
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

        print(f"[Celery] Done processing ZIP: {zip_s3_key}, total processed: {processed_count}")
        return {'status': 'completed', 'processed_count': processed_count}

    except Exception as e:
        db_session.rollback()
        print(f"[Celery] Error processing ZIP file: {e}")
        return {'status': 'failed', 'error': str(e)}

    finally:
        db_session.close()
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)
