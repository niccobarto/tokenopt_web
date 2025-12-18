# Utilizzo di RunPod con TokenOpt

Queste note spiegano come funziona il server FastAPI in `tokenopt_generator/server.py` e come collegarlo al sito Django tramite RunPod.

## Architettura in breve
- `tokenopt_generator/server.py` espone due endpoint FastAPI:
  - `GET /health` per il liveness check RunPod.
  - `POST /generate-inpainting` che accetta `prompt`, `num_generations`, `input_image` e `mask_image` come form-data, invoca `tto_web_generator.generate_inpainting` e restituisce le immagini come payload base64.
- `tokenopt_site/image_editor/services/generator.py` decide come generare:
  1. Se l'ENV `TOKENOPT_RUNPOD_URL` è valorizzato, invia la richiesta HTTP al server FastAPI su RunPod (`_generate_inpainting_runpod`).
  2. Se non c'è `TOKENOPT_RUNPOD_URL` ma `TOKENOPT_ENABLE_GPU=1`, usa la pipeline locale (`tto_web_generator.generate_inpainting`).
  3. In caso contrario usa il generatore "dummy" puramente CPU.

## Avviare il server su RunPod
1. **Costruisci l'immagine GPU** (include FastAPI e la pipeline):
   ```bash
   docker build -t tokenopt:gpu --build-arg APP_ENV=gpu .
   ```
2. **Esegui l'API** (esempio su porta 8001):
   ```bash
   uvicorn tokenopt_generator.server:app --host 0.0.0.0 --port 8001
   ```
   Variabili d'ambiente utili:
   - `TOKENOPT_ENABLE_GPU=1` per forzare la pipeline reale.
   - `TOKENOPT_USE_DUMMY_GENERATION=1` per test senza CUDA (ritorna immagini colorate fittizie).
3. **Verifica che il pod risponda**:
   ```bash
   curl http://<POD_HOST>:8001/health
   ```
4. **Esempio di chiamata inpainting**:
   ```bash
   curl -X POST "http://<POD_HOST>:8001/generate-inpainting" \
        -F "prompt=un gatto spaziale" \
        -F "num_generations=2" \
        -F "input_image=@/path/original.png" \
        -F "mask_image=@/path/mask.png"
   ```
   La risposta JSON contiene `results`, ognuno con `filename`, `content_type` e `data` (base64 PNG).

## Collegare il sito Django a RunPod
- Configura `TOKENOPT_RUNPOD_URL` (es. `http://<POD_HOST>:8001`).
- Lascia `TOKENOPT_ENABLE_GPU=0` nel web/worker: in questo modo `run_tto_job` userà sempre RunPod.
- I file generati vengono salvati in `TTO_JOBS_ROOT_ABSOLUTE/job_<id>/outputs` dal worker.

## Debug locale con Docker Compose
Il servizio `generator` nel `docker-compose.yml` avvia lo stesso server FastAPI sulla porta `8001` e il web/worker puntano a `http://generator:8001` tramite `TOKENOPT_RUNPOD_URL`. Puoi lanciare l'intero stack locale con:
```bash
docker compose up --build
```
Questo replica il flusso RunPod ma in locale.