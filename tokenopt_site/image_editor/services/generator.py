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


def _ensure_local_inputs(job: GenerationJob, base_dir: Path) -> Tuple[Path, Path]:
    """Scarica (se necessario) immagine e maschera nella workspace locale."""

    inputs_dir = base_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    input_image_path = inputs_dir / "original.png"
    mask_path = inputs_dir / "mask.png"

    if job.input_image and not input_image_path.exists():
        with job.input_image.open("rb") as src:
            input_image_path.write_bytes(src.read())

    if job.input_mask and not mask_path.exists():
        with job.input_mask.open("rb") as src:
            mask_path.write_bytes(src.read())

    return input_image_path, mask_path

def run_tto_job(job: GenerationJob) -> Sequence[Path]:
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
    outputs_dir.mkdir(parents=True, exist_ok=True)
    input_image_path, mask_path = _ensure_local_inputs(job, base_dir)
    generated_urls = []

    if RUNPOD_URL:
        generated_urls = _generate_inpainting_runpod(
            input_image_path=input_image_path,
            mask_path=mask_path,
            prompt=job.prompt,
            num_generations=job.num_generations,
            output_dir=outputs_dir,
            job=job,
        )
    elif os.getenv("TOKENOPT_ENABLE_GPU", "0") == "1":
        generated_urls = _generate_inpainting_local(
            input_image_path=base_dir / "inputs/original.png",
            mask_path=base_dir / "inputs/mask.png",
            prompt=job.prompt,
            num_generations=job.num_generations,
            output_dir=outputs_dir,
            job=job,
        )
    else:
        raise RuntimeError("Né RunPod né GPU locale disponibili per l'elaborazione")
    return [Path(p) for p in generated_urls]

def _generate_inpainting_local(
input_image_path: Path,
    mask_path: Path,
    prompt: str,
    num_generations: int,
    output_dir: Path,
    job: GenerationJob,
)-> List[Path]:
    from tokenopt_generator.api import tto_web_generator
    image_bytes=input_image_path.read_bytes()
    mask_bytes=mask_path.read_bytes()
    results=tto_web_generator.generate_inpainting_bytes(image_bytes,mask_bytes,prompt,num_generations)
    return _save_image_bytes(results,job.id)

def _generate_inpainting_runpod(
    input_image_path: Path,
    mask_path: Path,
    prompt: str,
    num_generations: int,
    output_dir: Path,
    job: GenerationJob,
) -> List[Path]:
    """Chiede a RunPod di generare le immagini e salva i risultati localmente."""

    endpoint = f"{RUNPOD_URL.rstrip('/')}/generate-inpainting"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "input_image": (input_image_path.name, input_image_path.open("rb"), "image/png"),
        "mask_image": (mask_path.name, mask_path.open("rb"), "image/png"),
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



def _save_image_bytes(results,job_id:int) -> List[Path]:
    out_paths: List[Path] = []
    for idx, result in enumerate(results, start=1):
        image_bytes = base64.b64decode(result["data"])
        file_name = f"{TTO_JOBS_ROOT_RELATIVE}/job_{job_id}/outputs/gen_{idx:02d}.png"
        saved_name = default_storage.save(file_name, ContentFile(image_bytes))
        url = default_storage.url(saved_name)
        out_paths.append(Path(url))
    return out_paths
