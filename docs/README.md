# Utilizzo di RunPod con TokenOpt

Queste note spiegano come funziona il server FastAPI in `tokenopt_generator/server.py` e come collegarlo al sito Django tramite RunPod.

## Architettura in breve
- `tokenopt_generator/server.py` espone endpoint FastAPI per tutte le operazioni GPU:
  - `GET /health` per il liveness check RunPod.
  - `POST /generate-inpainting` accetta `prompt`, `num_generations`, `input_image` e `mask_image` come form-data, invoca `tto_web_generator.generate_inpainting` e restituisce le immagini come payload base64.
  - `POST /remove-background` accetta `model` (es. `u2net`, `sam`, `isnet-general-use`, `birefnet-general`) e `input_image`, usa `rembg` e restituisce PNG base64 con alpha.
  - `POST /super-resolution` accetta `input_image` e richiama Real-ESRGAN via `TOKENOPT_SR_CLI_CMD` (o il default built-in) restituendo la PNG upscalata base64.
- I worker Django si appoggiano sempre al pod remoto se è configurato `TOKENOPT_RUNPOD_URL`:
  - `run_tto_job` usa `_generate_inpainting_runpod`.
  - `remove_background_task` invia l'immagine a `/remove-background`.
  - `run_super_resolution_task` invia l'immagine a `/super-resolution`.
  - Se `TOKENOPT_RUNPOD_URL` è vuoto, i worker usano le pipeline locali o la CLI definita in `SR_CLI_CMD`.

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
   - `TOKENOPT_SR_CLI_CMD` (facoltativa) per impostare il comando Real-ESRGAN del pod, es. JSON `"[\"realesrgan-ncnn-vulkan\", \"-i\", \"{in_path}\", \"-o\", \"{out_dir}\"]"` o stringa shell `"python image_editor/services/super_resolution.py -i {in_path} -o {out_dir} -n realesrgan-x4plus"`.
3. **Verifica che il pod risponda**:
   ```bash
   curl http://<POD_HOST>:8001/health
   ```
4. **Esempi di chiamata**:
   ```bash
   curl -X POST "http://<POD_HOST>:8001/generate-inpainting" \
        -F "prompt=un gatto spaziale" \
        -F "num_generations=2" \
        -F "input_image=@/path/original.png" \
        -F "mask_image=@/path/mask.png"
   ```
   Ritorna `results` con `filename`, `content_type` e `data` (base64 PNG).

   ```bash
   curl -X POST "http://<POD_HOST>:8001/remove-background" \
        -F "model=u2net" \
        -F "input_image=@/path/original.png"
   ```
   Ritorna `result.data` con PNG RGBA base64.

   ```bash
   curl -X POST "http://<POD_HOST>:8001/super-resolution" \
        -F "input_image=@/path/original.png"
   ```
   Ritorna `result.data` con PNG super-risoluta base64.

## Procedura completa su un pod nuovo
1. **Apri la shell del pod GPU**
   - Collegati via SSH/console RunPod alla VM del pod.

2. **Recupera il codice**
   - Clona il repository o scarica l'archivio nella home del pod:
     ```bash
     git clone <REPO_URL> && cd tokenopt_web
     ```

3. **Costruisci l'immagine**
   - Usa la Dockerfile GPU già configurata:
     ```bash
     docker build -t tokenopt:gpu --build-arg APP_ENV=gpu .
     ```

4. **Avvia il container dell'API**
   - Esempio minimale, esponendo la porta 8001 verso l'esterno del pod:
     ```bash
     docker run --rm -it --gpus all -p 8001:8001 \
       -e TOKENOPT_ENABLE_GPU=1 \
       tokenopt:gpu \
       uvicorn tokenopt_generator.server:app --host 0.0.0.0 --port 8001
     ```
   - Se vuoi solo testare la catena senza GPU, sostituisci `TOKENOPT_ENABLE_GPU=1` con `TOKENOPT_USE_DUMMY_GENERATION=1`.
   - Per super‑risoluzione personalizzata, aggiungi `-e "TOKENOPT_SR_CLI_CMD=[...]"` con il comando Real‑ESRGAN presente sul pod.

5. **Controlla lo stato**
   - Da una seconda shell (nello stesso pod o dall'esterno se la porta è esposta):
     ```bash
     curl http://<POD_HOST>:8001/health
     ```

6. **Prova gli endpoint**
   - Usa gli esempi `curl` sopra per inpainting, rimozione sfondo e super‑risoluzione.

7. **Collega Django al pod**
   - Nelle istanze web/worker (anche locali) imposta `TOKENOPT_RUNPOD_URL=http://<POD_HOST>:8001` e riavvia i servizi. Le task Celery chiameranno automaticamente il pod.

8. **Persistenza opzionale**
   - Se vuoi salvare log o cache tra riavvii, aggiungi volumi al `docker run`, ad esempio `-v /pod_cache:/root/.cache`.

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