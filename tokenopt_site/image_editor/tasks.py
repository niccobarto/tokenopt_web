# image_editor/tasks.py
from fileinput import filename

from celery import shared_task
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from tokenopt_site import settings
from tokenopt_site.settings import TTO_JOBS_ROOT_RELATIVE,REMOVEBG_ROOT_RELATIVE,SUPERRES_ROOT_RELATIVE
from .models import GenerationJob,SuperResolutionJob,RemoveBgJob
from .services.generator import run_tto_job
from tokenopt_generator.api.remove_background import remove_background
from tokenopt_generator.api.super_resolution import run_realesgan

@shared_task
def run_generation_task(job_id: int):
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
            file_name = f"{TTO_JOBS_ROOT_RELATIVE}/job_{job.id}/outputs/gen_{idx:02d}.png"
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
        out_bytes = remove_background(image_bytes,job.model_selected)

        # 4) salvo su storage SENZA uuid:
        #    uso job.id per avere un nome deterministico
        filename=f"{REMOVEBG_ROOT_RELATIVE}/job_{job.id}/outputs/output.png"

        # opzionale: se vuoi garantire overwrite, cancelli prima
        # if default_storage.exists(filename):
        #     default_storage.delete(filename)

        saved_path = default_storage.save(filename, ContentFile(out_bytes))

        # 5) aggancio l'output al model (ImageField)
        job.output_image= saved_path
        job.status = "COMPLETED"
        job.save(update_fields=["output_image", "status"])

        return True

    except Exception as e:
        job.status = "FAILED"
        job.error_message = str(e)
        job.save(update_fields=["status", "error_message"])
        return False

@shared_task
def run_super_resolution_task(job_id:int):
    job=SuperResolutionJob.objects.get(id=job_id)
    job.status="RUNNING"
    job.save(update_fields=["status"])

    try:
        with job.input_image.open("rb") as f:
            input_bytes=f.read()

        # Esegui SR black-box
        sr_cli_cmd = getattr(settings, "SR_CLI_CMD", None)
        if not sr_cli_cmd:
            raise RuntimeError("SR_CLI_CMD mancante in settings.py.")

        output_bytes=run_realesgan(input_bytes,sr_cli_cmd)
        file_path=f"{SUPERRES_ROOT_RELATIVE}/job_{job.id}/outputs/output.png"
        saved_path=default_storage.save(
            file_path,
            ContentFile(output_bytes),
        )
        job.superres_image=saved_path
        job.status="COMPLETED"
        job.save(update_fields=["status","superres_image"])
    except Exception as e:
        job.status = "FAILED"
        job.error_message = str(e)
        job.save()

