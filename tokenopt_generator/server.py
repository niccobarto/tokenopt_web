import base64
import json
import os
import shlex
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse


app = FastAPI()

@app.get("/health")
def health_check():
    """Semplice endpoint di health-check per RunPod."""

    return {"status": "ok"}


@app.post("/generate-inpainting")
async def generate_inpainting(
    prompt: str = Form(...),
    num_generations: int = Form(1),
    input_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
):
    from tokenopt_generator.api import tto_web_generator
    """Espone la pipeline di inpainting TTO via HTTP.

    L'endpoint viene pensato per girare su RunPod: riceve immagine e mask,
    richiama ``tto_web_generator.generate_inpainting`` e restituisce le immagini
    generate come base64.
    """

    try:
        num_generations = int(num_generations)
    except (TypeError, ValueError):
        num_generations = 1

    if num_generations < 1:
        raise HTTPException(status_code=400, detail="num_generations deve essere >= 1")

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        inputs_dir = base_dir / "inputs"
        outputs_dir = base_dir / "outputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        input_path = inputs_dir / "original.png"
        mask_path = inputs_dir / "mask.png"

        for upload, dest_path in (
            (input_image, input_path),
            (mask_image, mask_path),
        ):
            data = await upload.read()
            if not data:
                raise HTTPException(status_code=400, detail=f"{upload.filename} vuoto")
            dest_path.write_bytes(data)

        generated_paths = tto_web_generator.generate_inpainting(
            input_image_path=input_path,
            mask_path=mask_path,
            prompt=prompt,
            num_generations=num_generations,
            output_dir=outputs_dir,
        )

        results = []
        for path in generated_paths:
            image_bytes = Path(path).read_bytes()
            b64_data = base64.b64encode(image_bytes).decode("utf-8")
            results.append(
                {
                    "filename": Path(path).name,
                    "content_type": "image/png",
                    "data": b64_data,
                }
            )

        return JSONResponse({"results": results})


@app.post("/remove-background")
async def run_remove_background(
    model: str = Form("u2net"),
    input_image: UploadFile = File(...),
):
    """Esegue la rimozione dello sfondo via rembg e restituisce l'immagine base64."""
    from tokenopt_generator.api import remove_background
    image_bytes = await input_image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="input_image vuoto")

    output_bytes = remove_background.remove_background(image_bytes, model_selected=model)
    b64_data = base64.b64encode(output_bytes).decode("utf-8")

    return JSONResponse(
        {
            "result": {
                "filename": "removebg.png",
                "content_type": "image/png",
                "data": b64_data,
            }
        }
    )


def _resolve_sr_cli_cmd() -> list[str]:
    """
    Recupera il comando CLI per Real-ESRGAN dal valore d'ambiente ``TOKENOPT_SR_CLI_CMD``.

    Accetta sia JSON (lista) che una stringa space-separated. Se non valorizzato,
    usa un default compatibile con l'immagine RunPod.
    """

    env_value = os.getenv("TOKENOPT_SR_CLI_CMD", "").strip()
    if env_value:
        try:
            parsed = json.loads(env_value)
            if isinstance(parsed, list) and parsed:
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass

        split_cmd = shlex.split(env_value)
        if split_cmd:
            return split_cmd

    return [
        "python",
        "image_editor/services/super_resolution.py",
        "-i",
        "{in_path}",
        "-o",
        "{out_dir}",
        "-n",
        "realesrgan-x4plus",
    ]


@app.post("/super-resolution")
async def run_super_resolution(
    input_image: UploadFile = File(...),
):
    """Esegue Real-ESRGAN nel pod GPU e restituisce l'immagine upscalata in base64."""
    from tokenopt_generator.api import super_resolution
    image_bytes = await input_image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="input_image vuoto")

    sr_cli_cmd = _resolve_sr_cli_cmd()

    output_bytes = super_resolution.run_realesgan(image_bytes, sr_cli_cmd=sr_cli_cmd)
    b64_data = base64.b64encode(output_bytes).decode("utf-8")

    return JSONResponse(
        {
            "result": {
                "filename": "superres.png",
                "content_type": "image/png",
                "data": b64_data,
            }
        }
    )


__all__ = ["app"]