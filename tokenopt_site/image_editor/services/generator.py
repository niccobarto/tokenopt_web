# image_editor/services/tto_runner.py
import base64
import os
import time
from pathlib import Path
from typing import Sequence, List, Tuple
import requests
from PIL import Image, ImageDraw
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from image_editor.models import GenerationJob
from tokenopt_site.settings import TTO_JOBS_ROOT_RELATIVE

RUNPOD_URL = os.getenv("TOKENOPT_RUNPOD_URL", "").strip()
RUNPOD_TIMEOUT = int(os.getenv("TOKENOPT_RUNPOD_TIMEOUT", "600"))


def run_tto_job(job: GenerationJob) -> list[str]:
    """
    API principale:
    - usa l'id del job come identificatore
    - crea workspace
    - copia input
    - chiama generate_inpainting
    - ritorna i Path delle immagini generate
    """
    base_dir = Path(settings.TTO_JOBS_ROOT_ABSOLUTE) / f"job_{job.id}"
    outputs_dir = base_dir / "outputs"
    with job.input_image.open("rb") as f:
        input_image_bytes = f.read()

    with job.input_mask.open("rb") as f:
        mask_bytes = f.read()
    generated_urls = []

    if RUNPOD_URL:
        generated_urls = _generate_inpainting_runpod(
            input_image_bytes=input_image_bytes,
            mask_bytes=mask_bytes,
            prompt=job.prompt,
            num_generations=job.num_generations,
            job=job,
        )
    elif os.getenv("TOKENOPT_ENABLE_GPU", "0") == "1":
        generated_urls = _generate_inpainting_local(
            input_image_bytes=input_image_bytes,
            mask_bytes=mask_bytes,
            prompt=job.prompt,
            num_generations=job.num_generations,
            job=job,
        )
    else:
        raise RuntimeError("Né RunPod né GPU locale disponibili per l'elaborazione")
    return generated_urls

def _generate_inpainting_local(
    input_image_bytes: bytes,
    mask_bytes: bytes,
    prompt: str,
    num_generations: int,
    job: GenerationJob,
)-> List[str]:
    from tokenopt_generator.api import tto_web_generator
    results=tto_web_generator.generate_inpainting_bytes(input_image_bytes,mask_bytes,prompt,num_generations)
    return _save_image_bytes(results,job.id)

def _generate_inpainting_runpod(
    input_image_bytes: bytes,
    mask_bytes: bytes,
    prompt: str,
    num_generations: int,
    job: GenerationJob,
) -> List[str]:
    """Chiede a RunPod di generare le immagini e salva i risultati localmente."""

    endpoint = f"{RUNPOD_URL.rstrip('/')}/generate-inpainting"

    files = {
        "input_image": ("input.png", input_image_bytes, "image/png"),
        "mask_image": ("mask.png", mask_bytes, "image/png"),
    }
    data = {
        "prompt": prompt,
        "num_generations": str(num_generations),
    }

    response = requests.post(endpoint, data=data, files=files, timeout=RUNPOD_TIMEOUT)
    response.raise_for_status()
    job_id = response.json()["job_id"]

    payload = response.json()
    results = payload.get("results") or []

    if not isinstance(results, list):
        raise RuntimeError("Risposta RunPod malformata: 'results' non è una lista")

    status_url=f"{RUNPOD_URL.rstrip('/')}/jobs/{job_id}"
    while True:
        r=requests.get(status_url, timeout=10)
        r.raise_for_status()
        payload=r.json()

        if payload["status"]=="DONE":
            results=payload["result"]["results"]
            break
        if payload["status"]=="FAILED":
            raise RuntimeError(payload.get("error"))

        time.sleep(2)

    return _save_image_bytes(results,job.id)



def _save_image_bytes(results,job_id:int) -> List[str]:
    out_paths: List[str] = []
    for idx, result in enumerate(results, start=1):
        image_bytes = base64.b64decode(result["data"])
        file_name = f"{TTO_JOBS_ROOT_RELATIVE}/job_{job_id}/outputs/gen_{idx:02d}.png"
        saved_name = default_storage.save(file_name, ContentFile(image_bytes))
        url = default_storage.url(saved_name)
        out_paths.append(saved_name)
    return out_paths
