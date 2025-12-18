# image_editor/services/tto_runner.py
import base64
import os
from pathlib import Path
from typing import Sequence, List, Tuple
import requests
from PIL import Image, ImageDraw
from django.conf import settings
from image_editor.models import GenerationJob

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

    if RUNPOD_URL:
        generated_paths = _generate_inpainting_runpod(
            input_image_path=input_image_path,
            mask_path=mask_path,
            prompt=job.prompt,
            num_generations=job.num_generations,
            output_dir=outputs_dir,
        )
    elif os.getenv("TOKENOPT_ENABLE_GPU", "0") == "1":
        from tokenopt_generator.api import tto_web_generator
        generated_paths = tto_web_generator.generate_inpainting(
            input_image_path=base_dir / "inputs/original.png",
            mask_path=base_dir / "inputs/mask.png",
            prompt=job.prompt,
            num_generations=job.num_generations,
            output_dir=outputs_dir,
        )
    else:
        generated_paths = _generate_inpainting_dummy(
            input_image_path=input_image_path,
            mask_path=mask_path,
            prompt=job.prompt,
            num_generations=job.num_generations,
            output_dir=outputs_dir,
        )
    return [Path(p) for p in generated_paths]


def _generate_inpainting_runpod(
    input_image_path: Path,
    mask_path: Path,
    prompt: str,
    num_generations: int,
    output_dir: Path,
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

    payload = response.json()
    results = payload.get("results") or []

    if not isinstance(results, list):
        raise RuntimeError("Risposta RunPod malformata: 'results' non è una lista")

    out_paths: List[Path] = []
    for idx, result in enumerate(results, start=1):
        if "data" not in result:
            raise RuntimeError("Elemento di risposta RunPod privo di campo 'data'")

        filename = result.get("filename") or f"runpod_{idx:02d}.png"
        out_path = output_dir / filename
        image_bytes = base64.b64decode(result["data"])
        out_path.write_bytes(image_bytes)
        out_paths.append(out_path)

    return out_paths
def _generate_inpainting_dummy(
        input_image_path: Path,
        mask_path: Path,
        prompt: str,
        num_generations: int,
        output_dir: Path,
) -> List[Path]:
    """
    Generatore finto: NON usa torch, NON usa CUDA.
    Crea semplicemente dei quadrati colorati con un po' di testo.
    Serve solo per testare pipeline e salvataggio file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    out_paths: List[Path] = []

    # Carico l'immagine originale solo per prendere la size (se vuoi)
    try:
        base_img = Image.open(input_image_path).convert("RGB")
        width, height = base_img.size
    except Exception:
        # fallback se l'immagine non è leggibile
        width, height = 256, 256

    for i in range(num_generations):
        img = Image.new("RGB", (width, height), color=(200, 100 + 20 * i, 150))

        draw = ImageDraw.Draw(img)
        text = f"Dummy {i+1}\n{prompt[:20]}"
        draw.text((10, 10), text, fill=(0, 0, 0))

        out_path = output_dir / f"dummy_{i+1}.png"
        img.save(out_path)
        out_paths.append(out_path)

    return out_paths