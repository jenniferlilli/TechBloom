from celery import Celery
from db_model import DATABASE_URL
print(f"[Celery Worker] Using DATABASE_URL: {DATABASE_URL}")

def make_celery(app_name=__name__):
    celery = Celery(
        app_name,
        broker='redis://localhost:6379/0',
        backend=None 
    )
    celery.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
    )
    return celery
