# image_editor/tasks.py

from celery import shared_task
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from .models import GenerationJob,SuperResolutionJob
from .services.generator import run_tto_job
from .services.background import remove_background


@shared_task
def run_generation(job_id: int):
    job = GenerationJob.objects.get(id=job_id)

    try:
        job.status = "RUNNING"
        job.save(update_fields=["status"])

        # 1) Esegui la pipeline TTO (dummy o reale)
        generated_paths = run_tto_job(job)  # lista di Path locali

        # 2) Carica i file generati su R2
        generated_urls = []

        for idx, path in enumerate(generated_paths, start=1):
            with open(path, "rb") as f:
                data = f.read()

            file_name = f"outputs/job_{job.id}/gen_{idx:02d}.png"
            saved_name = default_storage.save(file_name, ContentFile(data))
            url = default_storage.url(saved_name)
            generated_urls.append(url)

        # 3) Aggiorna job
        job.status = "COMPLETED"
        job.generated_images = generated_urls
        job.save(update_fields=["status", "generated_images"])

    except Exception as e:
        job.status = "FAILED"
        job.error_message = str(e)
        job.save(update_fields=["status", "error_message"])
        raise


@shared_task
def process_super_resolution(job_id: int):
    job=SuperResolutionJob.objects.get(id=job_id)

    try:
        job.status="RUNNING"
        job.save(update_fields=["status"])

        #read input image
        with open(job.input_image.path, "rb") as f:
            data = f.read()

        file_name=f"superres/job_{job.id}/superres_image.png"
        saved_name=default_storage.save(file_name, ContentFile(data))
        url=default_storage.url(saved_name)

        job.superres_image=saved_name
        job.status="COMPLETED"
        job.save(update_fields=["status","superres_image"])
    except Exception as e:
        job.status="FAILED"
        job.error_message=str(e)
        job.save(update_fields=["status","error_message"])
        raise

@shared_task
def remove_background_task(image_bytes:bytes):
    out=remove_background(image_bytes)
    return out