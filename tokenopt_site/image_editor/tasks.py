# image_editor/tasks.py
import os
import tempfile
import requests
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from celery import shared_task
from django.conf import settings
from .models import GenerationJob

def download_to_temp(url: str) -> str:
    """
    Scarica un file da URL in una directory temporanea e ritorna il path locale.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(url)[1] or ".png")
    with os.fdopen(tmp_fd, "wb") as f:
        f.write(resp.content)
    return tmp_path


@shared_task
def run_generation(job_id: int):
    job = GenerationJob.objects.get(id=job_id)
    job.status = "RUNNING"
    job.save(update_fields=["status"])

    try:
        # 1) URL su R2
        input_image_url = job.input_image.url
        mask_url = job.input_mask.url if job.input_mask else None

        # 2) Scarico i file localmente sul worker
        local_input_path = download_to_temp(input_image_url)
        local_mask_path = download_to_temp(mask_url) if mask_url else None

        prompt = job.prompt
        num_generations = job.num_generations

        # 3) Chiamo la pipeline TTO che lavora su file locali
        #    run_tto deve ritornare una lista di oggetti PIL.Image (per esempio)
        from PIL import Image  # se ti serve qui

        generated_images_pil = run_tto(
            input_image_path=local_input_path,
            mask_path=local_mask_path,
            prompt=prompt,
            num_generations=num_generations,
        )

        # 4) Salvo ogni immagine PIL su R2 usando lo storage di Django
        from io import BytesIO

        generated_urls = []

        for idx, pil_img in enumerate(generated_images_pil, start=1):
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            buffer.seek(0)

            # nome nel bucket, es: outputs/job_42/gen_01.png
            file_name = f"outputs/job_{job.id}/gen_{idx:02d}.png"

            # default_storage usa R2MediaStorage -> file va su R2
            saved_name = default_storage.save(file_name, ContentFile(buffer.read()))
            url = default_storage.url(saved_name)
            generated_urls.append(url)

        # 5) Aggiorno il job
        job.status = "COMPLETED"
        job.generated_images = generated_urls
        job.save(update_fields=["status", "generated_images"])

    except Exception as e:
        job.status = "FAILED"
        job.error_message = str(e)
        job.save(update_fields=["status", "error_message"])
        raise
