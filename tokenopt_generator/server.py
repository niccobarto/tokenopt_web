# python
from fastapi import FastAPI
from pydantic import BaseModel
# from tokenopt_generator.services.generator import run_tto_job  # usa la tua funzione reale

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    params: dict | None = None

@app.post("/generate")
def generate(req: GenerateRequest):
    # outputs = run_tto_job(prompt=req.prompt, params=req.params)
    outputs = [{"text": f"Generato: {req.prompt}"}]  # placeholder
    return {"results": outputs}