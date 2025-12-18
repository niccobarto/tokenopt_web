import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from tokenopt_generator.api import tto_web_generator
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


__all__ = ["app"]