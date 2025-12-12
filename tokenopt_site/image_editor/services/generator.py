# image_editor/services/tto_runner.py
import os
from pathlib import Path
import json
import shutil
from typing import Sequence, List

from PIL import Image, ImageDraw
from django.conf import settings
from image_editor.models import GenerationJob

def create_workspace_for_job(job_id: int) -> Path:
    """
    Crea la cartella di lavoro per questo job.

    Struttura:
      media/tto_jobs/user_<user_id>/job_<id>/
        inputs/
        outputs/
        logs/
    Se l'utente non c'è, salta la parte user_<user_id>.
    """
    base_root = Path(settings.TTO_JOBS_ROOT)
    base_dir = base_root / f"job_{job_id}"

    (base_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (base_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (base_dir / "logs").mkdir(parents=True, exist_ok=True)

    return base_dir


def save_inputs_to_workspace(
    job: GenerationJob,
    base_dir: Path,
) -> dict:
    """
    Copia input nel workspace e salva params.json
    """
    inputs_dir = base_dir / "inputs"

    img_dest = inputs_dir / "original.png"
    mask_dest = inputs_dir / "mask.png"

    # Copia immagine di input dalla storage backend al filesystem locale
    with job.input_image.open("rb") as src, open(img_dest, "wb") as dst:
        shutil.copyfileobj(src, dst)

    # Copia maschera se presente
    if job.input_mask:
        with job.input_mask.open("rb") as src, open(mask_dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        mask_dest = None

    params = {
        "prompt": job.prompt,
        "num_generations": job.num_generations,
        "job_id": job.id,
    }
    with open(inputs_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    return {
        "original_img": img_dest,
        "mask": mask_dest,
        "params": params,
    }


def run_tto_job(job: GenerationJob) -> Sequence[Path]:
    """
    API principale:
    - usa l'id del job come identificatore
    - crea workspace
    - copia input
    - chiama generate_inpainting
    - ritorna i Path delle immagini generate
    """


    # 1. workspace
    base_dir = create_workspace_for_job(job.id)
    outputs_dir = base_dir / "outputs"

    # 2. copia input
    paths_info = save_inputs_to_workspace(job, base_dir)
    generated_paths=[]
    if os.getenv("TOKENOPT_ENABLE_GPU", "0") != "1":
        generated_paths=_generate_inpainting_dummy(
            input_image_path=paths_info["original_img"],
            mask_path=paths_info["mask"],
            prompt=job.prompt,
            num_generations=job.num_generations,
            output_dir=outputs_dir,
        )
    else:
        from tokenopt_generator.generator import tto_web_generator
        # 3. chiama il generatore vero e proprio (tokenopt_generator)
        generated_paths = tto_web_generator.generate_inpainting(
            input_image_path=paths_info["original_img"],
            mask_path=paths_info["mask"],
            prompt=job.prompt,
            num_generations=job.num_generations,
            output_dir=outputs_dir,
        )

    #lista di Path
    return [Path(p) for p in generated_paths]

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