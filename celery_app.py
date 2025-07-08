from celery import Celery
from db_model import DATABASE_URL
print(f"[Celery Worker] Using DATABASE_URL: {DATABASE_URL}")

def make_celery():
    celery = Celery(
        'tasks',
        broker=os.environ.get('CELERY_BROKER_URL'),
        backend=os.environ.get('CELERY_RESULT_BACKEND')
    )
    celery.conf.update(
        task_soft_time_limit=60,
        task_time_limit=90,
        worker_concurrency=4,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
    )
    return celery
