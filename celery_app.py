import os
from celery import Celery
from db_model import DATABASE_URL

print(f"[Celery Worker] Using DATABASE_URL: {DATABASE_URL}")

def make_celery():
    broker_url = os.environ.get('CELERY_BROKER_URL')
    backend_url = os.environ.get('CELERY_RESULT_BACKEND')

    if not broker_url:
        raise ValueError("CELERY_BROKER_URL environment variable not set")
    # backend can be optional if you don't use results

    celery = Celery(
        'tasks',
        broker=broker_url,
        backend=backend_url
    )

    celery.conf.update(
        task_soft_time_limit=300,
        task_time_limit=360,
        worker_concurrency=4,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
    )
    return celery
