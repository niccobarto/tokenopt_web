# image_editor/services/tto_runner.py
import os
from pathlib import Path
import json
import shutil
from typing import Sequence, List

from PIL import Image, ImageDraw
from django.conf import settings
from image_editor.models import GenerationJob


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

    generated_paths=[]
    if os.getenv("TOKENOPT_ENABLE_GPU", "0") != "1":
        generated_paths=_generate_inpainting_dummy(
            input_image_path=base_dir /"inputs/original.png",
            mask_path=base_dir/"inputs/mask.png",
            prompt=job.prompt,
            num_generations=job.num_generations,
            output_dir=outputs_dir,
        )
    else:
        from tokenopt_generator.api import tto_web_generator
        # 3. chiama il generatore vero e proprio (tokenopt_generator)
        generated_paths = tto_web_generator.generate_inpainting(
            input_image_path=base_dir/"inputs/original.png",
            mask_path=base_dir/"inputs/mask.png",
            prompt=job.prompt,
            num_generations=job.num_generations,
            output_dir=outputs_dir,
        )

    #lista di Path
    for p in generated_paths:
        print(p)
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