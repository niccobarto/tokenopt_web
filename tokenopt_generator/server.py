import base64
import json
import threading
import os
import shlex
import tempfile
from pathlib import Path
import uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse


app = FastAPI()
JOBS={}

@app.get("/health")
def health_check():
    """Semplice endpoint di health-check per RunPod."""

    return {"status": "ok"}

def _run_job(job_id: str, prompt: str, num_generations: int, input_bytes: bytes, mask_bytes: bytes):
    from tokenopt_generator.api import tto_web_generator
    try:
        JOBS[job_id]["status"] = "RUNNING"
        results = tto_web_generator.generate_inpainting_bytes(
            input_image_bytes=input_bytes,
            input_mask_bytes=mask_bytes,
            prompt=prompt,
            num_generations=num_generations,
        )

        # serializziamo in base64 SOLO QUI (boundary di rete)
        encoded = []
        for r in results:
            encoded.append({
                "filename": r["filename"],
                "content_type": r["content_type"],
                "data": base64.b64encode(r["data"]).decode("utf-8"),
            })

        JOBS[job_id]["status"] = "DONE"
        JOBS[job_id]["result"] = {"results": encoded}
    except Exception as e:
        JOBS[job_id]["status"] = "FAILED"
        JOBS[job_id]["error"] = str(e)

@app.post("/generate-inpainting")
async def generate_inpainting(
    prompt: str = Form(...),
    num_generations: int = Form(1),
    input_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
):
    """Espone la pipeline di inpainting TTO via HTTP.

    L'endpoint viene pensato per girare su RunPod: riceve immagine e mask,
    richiama ``tto_web_generator.generate_inpainting`` e restituisce le immagini
    generate come base64.
    """

    input_bytes = await input_image.read()
    mask_bytes = await mask_image.read()
    if not input_bytes or not mask_bytes:
        raise HTTPException(status_code=400, detail="input/mask vuoti")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "QUEUED", "result": None, "error": None}

    th = threading.Thread(
        target=_run_job,
        args=(job_id, prompt, int(num_generations), input_bytes, mask_bytes),
        daemon=True,
    )
    th.start()

    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "QUEUED"})

@app.get("/jobs/{job.id}") # Recupera lo stato di un job
def job_status(job_id:str):
    job=JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job non trovato")
    return JSONResponse(content=job)

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