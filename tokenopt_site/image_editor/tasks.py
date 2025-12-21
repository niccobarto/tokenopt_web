# image_editor/tasks.py
from fileinput import filename

from celery import shared_task
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

import base64
import os

import requests
from tokenopt_site.settings import TTO_JOBS_ROOT_RELATIVE,REMOVEBG_ROOT_RELATIVE,SUPERRES_ROOT_RELATIVE
from .models import GenerationJob,SuperResolutionJob,RemoveBgJob
from .services.generator import run_tto_job
from tokenopt_generator.api.remove_background import remove_background

RUNPOD_URL = os.getenv("TOKENOPT_RUNPOD_URL", "").strip()
RUNPOD_TIMEOUT = int(os.getenv("TOKENOPT_RUNPOD_TIMEOUT", "600"))


@shared_task
def run_generation_task(job_id: int):
    job = GenerationJob.objects.get(id=job_id)

    try:
        job.status = "RUNNING"
        job.save(update_fields=["status"])

        generated_urls = run_tto_job(job)

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
        if RUNPOD_URL:
            out_bytes = _remove_background_runpod(image_bytes, job.model_selected)
        else:
            out_bytes = remove_background(image_bytes, job.model_selected)
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

def _remove_background_runpod(image_bytes: bytes, model_selected: str) -> bytes:
    endpoint = f"{RUNPOD_URL.rstrip('/')}/remove-background"

    files = {"input_image": ("input.png", image_bytes, "image/png")}
    data = {"model": model_selected}

    response = requests.post(endpoint, data=data, files=files, timeout=RUNPOD_TIMEOUT)
    response.raise_for_status()

    payload = response.json()
    result = payload.get("result")
    if not result or "data" not in result:
        raise RuntimeError("Risposta RunPod remove-background priva del campo 'data'")

    return base64.b64decode(result["data"])

@shared_task
def run_super_resolution_task(job_id:int):
    job=SuperResolutionJob.objects.get(id=job_id)
    job.status="RUNNING"
    job.save(update_fields=["status"])

    try:
        # Leggo l'immagine di input salvata
        with job.input_image.open("rb") as f:
            input_bytes=f.read()

        # Richiamo sempre il servizio remoto di super-resolution
        if not RUNPOD_URL:
            raise RuntimeError("TOKENOPT_RUNPOD_URL non configurato per la super-resolution.")

        output_bytes = _super_resolution_runpod(input_bytes)

        file_path=f"{SUPERRES_ROOT_RELATIVE}/job_{job.id}/outputs/output.png"
        saved_path = default_storage.save(file_path, ContentFile(output_bytes))

        job.superres_image=saved_path
        job.status="COMPLETED"
        job.save(update_fields=["status","superres_image"])
    except Exception as e:
        job.status = "FAILED"
        job.error_message = str(e)
        job.save(update_fields=["status", "error_message"])
        return False

    return True


def _super_resolution_runpod(image_bytes: bytes) -> bytes:
    endpoint = f"{RUNPOD_URL.rstrip('/')}/super-resolution"

    files = {"input_image": ("input.png", image_bytes, "image/png")}

    response = requests.post(endpoint, files=files, timeout=RUNPOD_TIMEOUT)
    response.raise_for_status()

    payload = response.json()
    result = payload.get("result")
    if not result or "data" not in result:
        raise RuntimeError("Risposta RunPod super-resolution priva del campo 'data'")

    return base64.b64decode(result["data"])