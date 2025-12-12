# image_editor/services/tto_runner.py
from pathlib import Path
import json
import shutil
from typing import Sequence

from django.conf import settings
from image_editor.models import GenerationJob

# importa la funzione che hai definito nella repo di generazione
from tokenopt_generator.generator import tto_web_generator


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
