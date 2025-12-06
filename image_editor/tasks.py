# image_editor/tasks.py
from celery import shared_task
from django.conf import settings
from .models import GenerationJob
import os

@shared_task
def run_generation(job_id: int):
    job = GenerationJob.objects.get(id=job_id)
    job.status = "RUNNING"
    job.save(update_fields=["status"])

    try:
        input_image_path = job.input_image.path
        mask_path = job.input_mask.path if job.input_mask else None

        prompt = job.prompt
        num_generations = job.num_generations

        # Cartella base di output (es. media/outputs/)
        base_output_dir = os.path.join(settings.MEDIA_ROOT, "outputs")
        os.makedirs(base_output_dir, exist_ok=True)

        # Cartella specifica per questo job
        job_output_dir = os.path.join(base_output_dir, f"job_{job.id}")
        os.makedirs(job_output_dir, exist_ok=True)

        # TODO: qui dentro chiamerai la tua pipeline TTO
        # es: generated_paths = run_tto(input_image_path, mask_path, prompt, num_generations, job_output_dir)

        generated_paths = []  # placeholder

        job.status = "COMPLETED"
        job.output_dir = job_output_dir

        # Se HAI il campo generated_images nel modello:
        # job.generated_images = generated_paths
        # job.save(update_fields=["status", "output_dir", "generated_images"])

        # Se NON hai il campo generated_images:
        job.save(update_fields=["status", "output_dir"])

    except Exception as e:
        job.status = "FAILED"
        job.error_message = str(e)
        job.save(update_fields=["status", "error_message"])
        raise
