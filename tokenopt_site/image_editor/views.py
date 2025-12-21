import base64
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST

from .models import GenerationJob,RemoveBgJob
from .services.generator import run_tto_job
from .tasks import run_generation_task,remove_background_task,run_super_resolution_task

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

@require_POST
@csrf_protect
def create_upload(request):
    uploaded_image=request.FILES.get("image")
    if uploaded_image is None:
        return JsonResponse({"ok": False, "error": "File 'image' mancante"}, status=400)
    upload=UserUpload.objects.create(original_image=uploaded_image)
    return JsonResponse({"ok": True, "upload_id": upload.id})


def dataurl_to_contentfile(dataurl: str, filename: str) -> ContentFile | None:
    """
    Converte una stringa dataURL (es. 'data:image/png;base64,...')
    in un ContentFile Django con un nome di file.

    NOTA: con DEFAULT_FILE_STORAGE puntato a R2, quando questo ContentFile
    viene assegnato a un ImageField, Django lo carica automaticamente
    su Cloudflare R2 (non sul disco locale).
    """
    if not dataurl:
        return None

    try:
        header, b64data = dataurl.split(",", 1)
    except ValueError:
        return None

    # header esempio: "data:image/png;base64"
    mime_part = header.split(";")[0]     # "data:image/png"
    ext = mime_part.split("/")[-1]       # "png"

    try:
        img_bytes = base64.b64decode(b64data)
    except base64.binascii.Error:
        return None

    return ContentFile(img_bytes, name=f"{filename}.{ext}")

def home_view(request):
    # GET: mostra pagina vuota / stato iniziale
    return render(
        request,
        "image_editor/index.html",
        {
            "last_prompt": "",
            "last_num_generations":"1",
            "errors": [],
        },
    )

@require_POST
@csrf_protect
def start_generation_view(request):
    """
    - POST: riceve gli input (prompt, immagine, maschera),
            crea un GenerationJob e lancia il task Celery.
    Le immagini vengono salvate su R2 automaticamente grazie a DEFAULT_FILE_STORAGE.
    """
    prompt = request.POST.get("prompt", "").strip()
    num_generations = request.POST.get("num_generations", "1").strip()
    original_dataurl = request.POST.get("original_image", "").strip()
    mask_dataurl = request.POST.get("mask_image", "").strip()
    upload_id = request.POST.get("upload_id")
    configs = {
        "config1": bool(request.POST.get("config1")),
        "config2": bool(request.POST.get("config2")),
        "config3": bool(request.POST.get("config3")),
        "config4": bool(request.POST.get("config4")),
    }

    upload=None
    if upload_id:
        try:
            upload = UserUpload.objects.get(id=int(upload_id))
        except (ValueError, UserUpload.DoesNotExist):
            upload = None
    # Validazione lato server parallela a JS
    errors = []
    if not prompt:
        errors.append("Il prompt non può essere vuoto.")

    try:
        num_generations = int(num_generations)
    except ValueError:
        num_generations = 1
        errors.append("Numero di generazioni non valido, uso 1.")

    if num_generations < 1 or num_generations > 4:
        num_generations = 1
        errors.append("Numero di generazioni deve essere tra 1 e 4, uso 1.")

    # Decodifica delle immagini da dataURL -> ContentFile
    original_img = dataurl_to_contentfile(original_dataurl, "original")
    mask_img = dataurl_to_contentfile(mask_dataurl, "mask")

    if original_img is None:
        errors.append("Immagine originale non valida.")
    if mask_img is None:
        errors.append("Immagine maschera non valida.")

    if errors:
        return render(request, "image_editor/index.html", {
            "generated_images": None,
            "errors": errors,
            "last_prompt": prompt,
            "last_num_generations": num_generations,
        })

    # QUI: quando salvi il model, input_image/input_mask vanno su R2
    job = GenerationJob.objects.create(
        upload=upload,
        prompt=prompt,
        num_generations=num_generations,
        status="PENDING",
        configs=configs,
    )
    job.input_mask=mask_img
    job.input_image=original_img
    job.save()
     # Avvia Celery — asincrono
    run_generation_task.delay(job.id)

    return JsonResponse({"ok": True, "job_id": job.id})


def generation_status_view(request, job_id: int):
    """
    View per controllare lo stato di un job di generazione.

    Ritorna JSON con:
    - status: PENDING / RUNNING / COMPLETED / FAILED
    - error: eventuale messaggio di errore
    - urls: lista di URL (su R2) delle immagini generate
    """
    job = get_object_or_404(GenerationJob, id=job_id)
    urls=[]
    for img_url in job.generated_images:
        if not img_url.__contains__("media"):
            urls.append(default_storage.url(img_url))
        else:
            urls.append(img_url)
    data = {
        "job_id": job.id,
        "status": job.status,
        "error": job.error_message,
        "generated_images": urls,
        # volendo puoi anche inserire gli URL di input:
        # "input_image": job.input_image.url if job.input_image else None,
        # "input_mask": job.input_mask.url if job.input_mask else None,
    }
    return JsonResponse(data)


ALLOWED_MODELS=["u2net","sam","isnet-general-use","birefnet-general"]
@require_POST
@csrf_protect
def remove_background(request):
    """
    Riceve un file immagine, crea un RemoveBgJob e avvia il task Celery.
    Ritorna subito job_id per polling (NON blocca).
    """
    uploaded_image = request.FILES.get("image")
    model_selected=request.POST.get("model")

    # L’immagine viene inviata come file (multipart/form-data) perché è un dato binario:
    # Django separa automaticamente i campi testuali (request.POST) dai file (request.FILES).
    # Questo evita l’uso di Base64/JSON, riduce overhead di memoria e consente un parsing
    # corretto ed efficiente dei contenuti binari secondo lo standard HTTP.

    upload_id = request.POST.get("upload_id")
    upload = None
    if upload_id:
        try:
            upload = UserUpload.objects.get(id=int(upload_id))
        except (ValueError, UserUpload.DoesNotExist):
            upload = None

    if model_selected not in ALLOWED_MODELS:
        return JsonResponse({"ok": False, "error": "Modello non valido"}, status=400)

    if uploaded_image is None:
        return JsonResponse({"ok": False, "error": "File 'image' mancante"}, status=400)

    # Creo job e salvo input (va su R2 se configurato)
    job = RemoveBgJob.objects.create(
        upload=upload,
        status="PENDING",
        model_selected=model_selected,
    )
    job.input_image=uploaded_image
    job.save()
    # Avvio task asincrono
    remove_background_task.delay(job.id)

    # Risposta coerente con start_generation_view
    return JsonResponse({"ok": True, "job_id": job.id, "status": "PENDING"})

def remove_background_result_view(request, job_id: int): #Utilizzata per servire l'immagine risultante in "same-origin"
    """
    Serve l'immagine risultante (background rimosso) come risorsa SAME-ORIGIN.

    Motivo:
    - se il frontend carica direttamente da R2, il browser blocca per CORS.
    - servendo da Django (localhost:8000) il canvas può usare drawImage senza problemi.
    """
    job = get_object_or_404(RemoveBgJob, id=job_id)

    # Se il job non è pronto, ritorno 404 JSON (utile anche per debug)
    if job.status != "COMPLETED" or not job.output_image:
        return JsonResponse({"ok": False, "error": "Risultato non disponibile"}, status=404)

    # Leggo il file dallo storage (R2 o locale)
    output_image=default_storage.open(job.output_image).read()

    # Ritorno i bytes come immagine PNG
    response = HttpResponse(output_image, content_type="image/png")

    # Evita cache aggressiva durante sviluppo
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["Pragma"] = "no-cache"

    return response


def remove_background_status_view(request, job_id: int):
    """
    Polling endpoint:
    ritorna status e, quando COMPLETED, anche image_url.
    """
    job = get_object_or_404(RemoveBgJob, id=job_id)

    image_url = None
    if job.status == "COMPLETED" and job.output_image:
        image_url = job.output_image

    return JsonResponse({
        "job_id": job.id,
        "status": job.status,
        "error": job.error_message,
        "image_url": image_url,
    })

import time
import requests

from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect
from django.core.files.base import ContentFile

from .models import UserUpload, SuperResolutionJob
from .tasks import run_super_resolution_task


@require_POST
@csrf_protect
def start_superres(request):
    """
    Crea un SuperResolutionJob a partire da:
      - image_url: URL pubblico dell'immagine generata
      - upload_id: (opzionale) FK a UserUpload per mantenere il workflow collegato

    Ritorna:
      - ok, job_id
    """
    image_url = (request.POST.get("image_url") or "").strip()
    upload_id = (request.POST.get("upload_id") or "").strip()

    if not image_url:
        return JsonResponse({"ok": False, "error": "image_url mancante."}, status=400)

    # Recupero upload se presente
    upload = None
    if upload_id:
        try:
            upload = UserUpload.objects.get(id=int(upload_id))
        except (ValueError, UserUpload.DoesNotExist):
            upload = None

    # Creo job PENDING (senza input_image, lo setto subito dopo)
    job = SuperResolutionJob.objects.create(
        upload=upload,
        status="PENDING",
    )

    try:
        # Scarico l'immagine (URL) e la salvo nell'ImageField input_image
        r = requests.get(image_url, timeout=30)
        r.raise_for_status()

        ts = int(time.time())
        filename = f"original.png"  # niente uuid

        job.input_image.save(filename, ContentFile(r.content), save=True)

    except Exception as e:
        job.status = "FAILED"
        job.error_message = f"Errore download/salvataggio input_image: {e}"
        job.save(update_fields=["status", "error_message"])
        return JsonResponse({"ok": False, "error": job.error_message}, status=500)

    # Avvio task asincrono
    run_super_resolution_task.delay(job.id)

    return JsonResponse({"ok": True, "job_id": job.id})


def superres_status(request, job_id: int):
    """
    Endpoint di polling:
    GET /editor/superres_status/<job_id>/

    Ritorna sempre:
      - status
    Opzionale:
      - output_url (se COMPLETED)
      - error (se FAILED)
    """
    job = get_object_or_404(SuperResolutionJob, id=job_id)

    data = {
        "status": job.status
    }

    if job.status == "FAILED":
        data["error"] = job.error_message or "Errore sconosciuto."

    if job.status == "COMPLETED" and job.superres_image:
        data["output_url"] = default_storage.url(job.superres_image)

    return JsonResponse(data)