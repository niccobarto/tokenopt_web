# image_editor/tasks.py

from celery import shared_task
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from .models import GenerationJob,SuperResolutionJob,RemoveBgJob
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
def remove_background_task(job_id: int):
    """
    Task asincrono:
    - marca RUNNING
    - legge bytes dell'immagine input dallo storage
    - calcola output bytes PNG (sfondo rimosso)
    - salva output su storage
    - marca COMPLETED (o FAILED con error_message)
    """
    job = RemoveBgJob.objects.get(id=job_id)

    # 1) set RUNNING
    job.status = "RUNNING"
    job.error_message = ""
    job.save(update_fields=["status", "error_message"])

    try:
        # 2) leggo input bytes (funziona con R2 o filesystem)
        with job.input_image.open("rb") as f:
            image_bytes = f.read()

        # 3) eseguo la rimozione dello sfondo (ritorna bytes PNG)
        out_bytes = remove_background(image_bytes)

        # 4) salvo su storage SENZA uuid:
        #    uso job.id per avere un nome deterministico
        filename = f"remove_bg/outputs/job_{job.id}.png"

        # opzionale: se vuoi garantire overwrite, cancelli prima
        # if default_storage.exists(filename):
        #     default_storage.delete(filename)

        saved_path = default_storage.save(filename, ContentFile(out_bytes))

        # 5) aggancio l'output al model (ImageField)
        job.output_image.name = saved_path
        job.status = "COMPLETED"
        job.save(update_fields=["output_image", "status"])

        return True

    except Exception as e:
        job.status = "FAILED"
        job.error_message = str(e)
        job.save(update_fields=["status", "error_message"])
        return False