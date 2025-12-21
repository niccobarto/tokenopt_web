import base64
import json
import threading
import os
import shlex
import uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
JOBS={}

@app.get("/health")
def health_check():
    """Semplice endpoint di health-check per RunPod."""

    return {"status": "ok"}

def _run_job(job_id: str, prompt: str, num_generations: int, configs:dict[str,bool] ,input_bytes: bytes, mask_bytes: bytes):
    from tokenopt_generator.api import tto_web_generator
    try:
        JOBS[job_id]["status"] = "RUNNING"
        results = tto_web_generator.generate_inpainting_bytes(
            input_image_bytes=input_bytes,
            input_mask_bytes=mask_bytes,
            prompt=prompt,
            num_generations=num_generations,
            configs=configs,
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
    configs: str = Form("{}"),
    input_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
):
    """Espone la pipeline di inpainting TTO via HTTP.

    L'endpoint viene pensato per girare su RunPod: riceve immagine e mask,
    richiama ``tto_web_generator.generate_inpainting`` e restituisce le immagini
    generate come base64.
    """

    try:
        configs_dict = json.loads(configs)
        configs_bool: dict[str, bool] = {k: bool(v) for k, v in configs_dict.items()}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="configs non è JSON valido")

    if not isinstance(configs_dict, dict):
        raise HTTPException(status_code=400, detail="configs deve essere un oggetto JSON")

    input_bytes = await input_image.read()
    mask_bytes = await mask_image.read()
    if not input_bytes or not mask_bytes:
        raise HTTPException(status_code=400, detail="input/mask vuoti")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "QUEUED", "result": None, "error": None}

    th = threading.Thread(
        target=_run_job,
        args=(job_id, prompt, int(num_generations), configs_bool, input_bytes, mask_bytes),
        daemon=True,
    )
    th.start()

    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "QUEUED"})

@app.get("/jobs/{job_id}") # Recupera lo stato di un job
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

@app.post("/super-resolution")
async def run_super_resolution(
    input_image: UploadFile = File(...),
):
    """
    Esegue Real-ESRGAN su CUDA (PyTorch) e restituisce PNG base64.
    """
    from tokenopt_generator.api.super_resolution import run_realesgan, SRConfig

    image_bytes = await input_image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="input_image vuoto")

    # Config: puoi anche leggerla da env se vuoi (ma non è necessario)
    cfg = SRConfig(
        weights_path="/workspace/models/realesrgan/RealESRGAN_x4plus.pth",
        tile=0,         # se OOM, metti 256 o 512
        half=True,
        device="cuda",
    )

    try:
        output_bytes = run_realesgan(image_bytes, cfg=cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
