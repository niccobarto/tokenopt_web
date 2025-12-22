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
import json
from image_editor.models import GenerationJob
from tokenopt_site.settings import TTO_JOBS_ROOT_RELATIVE

RUNPOD_URL = os.getenv("TOKENOPT_RUNPOD_URL", "").strip()
RUNPOD_TIMEOUT = int(os.getenv("TOKENOPT_RUNPOD_TIMEOUT", "600"))
DUMMY_GENERATION = True if os.getenv("DUMMY_GENERATION") == "True" else False

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

    if RUNPOD_URL and not DUMMY_GENERATION:
        generated_urls = _generate_inpainting_runpod(
            input_image_bytes=input_image_bytes,
            mask_bytes=mask_bytes,
            prompt=job.prompt,
            num_generations=job.num_generations,
            configs=job.configs,
            job=job,
        )
    else:
        generated_urls = _generate_inpainting_local(
            input_image_bytes=input_image_bytes,
            mask_bytes=mask_bytes,
            prompt=job.prompt,
            num_generations=job.num_generations,
            configs=job.configs,
            job=job,
            dummy=DUMMY_GENERATION
        )
    return generated_urls


def _generate_inpainting_local(
    input_image_bytes: bytes,
    mask_bytes: bytes,
    prompt: str,
    num_generations: int,
    configs:dict[str,bool],
    job: GenerationJob,
    dummy:bool,
)-> List[str]:
    if not dummy:
        from tokenopt_generator.api import tto_web_generator
        results=tto_web_generator.generate_inpainting_bytes(input_image_bytes,mask_bytes,prompt,num_generations,configs)
        return _save_image_bytes(results,job.id)
    else:
        return _generate_inpainting_dummy(
            input_image_bytes=input_image_bytes,
            input_mask_bytes=mask_bytes,
            prompt=prompt,
            num_generations=num_generations,
            configs=configs,
            output_dir= Path(f"{TTO_JOBS_ROOT_RELATIVE}/job_{job.id}/outputs"),
        )


def _generate_inpainting_runpod(
    input_image_bytes: bytes,
    mask_bytes: bytes,
    prompt: str,
    num_generations: int,
    job: GenerationJob,
    configs:dict[str,bool],
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
        "configs": json.dumps(configs),
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

def _generate_inpainting_dummy(
        input_image_bytes: bytes,
        input_mask_bytes: bytes,
        prompt: str,
        num_generations: int,
        configs: dict[str, bool],
        output_dir: Path,
) -> List[str]:
    """
    Generatore finto: salva usando lo storage Django.
    Nota: `output_dir` deve essere relativo alla MEDIA_ROOT (non MEDIA_URL).
    Ritorna i path salvati (relativi allo storage); per URL usare `default_storage.url(saved_name)`.
    """
    # Assicuriamoci che output_dir sia relativo allo storage (MEDIA_ROOT)
    # e che la directory esista: default_storage si occupa della creazione.

    out_paths: List[str] = []

    # Determina dimensioni dai bytes
    try:
        from PIL import Image
        import io
        base_img = Image.open(io.BytesIO(input_image_bytes)).convert("RGB")
        width, height = base_img.size
    except Exception:
        width, height = 256, 256

    num_configs=0
    for k,v in configs.items():
        if v:
            num_configs += 1
    # Genera e salva tramite storage
    for idx in range(num_configs):
        img = Image.new("RGB", (width, height), color=(200, 100 + 20 * idx, 150))
        draw = ImageDraw.Draw(img)
        text = f"gen_{idx+1:02d}\n{prompt[:32]}"
        draw.text((10, 10), text, fill=(0, 0, 0))

        rel_file = f"{output_dir}/gen_{idx+1:02d}.png"
        # Serializza in memoria e salva via storage (crea cartella se manca)
        import io as _io
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        saved_name = default_storage.save(rel_file, ContentFile(buf.read()))
        out_paths.append(saved_name)

    return out_paths