import base64
import uuid

from PIL.Image import Image
from django.core.files.base import ContentFile
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST

from .models import GenerationJob,RemoveBgJob
from .services.generator import run_tto_job
from .tasks import run_generation,remove_background_task

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile




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


def start_generation_view(request):
    """
    - GET: mostra la pagina dell'editor di maschera
    - POST: riceve gli input (prompt, immagine, maschera),
            crea un GenerationJob e lancia il task Celery.

    Le immagini vengono salvate su R2 automaticamente grazie a DEFAULT_FILE_STORAGE.
    """
    if request.method == "POST":
        prompt = request.POST.get("prompt", "").strip()
        num_generations = request.POST.get("num_generations", "1").strip()
        original_dataurl = request.POST.get("original_image", "").strip()
        mask_dataurl = request.POST.get("mask_image", "").strip()

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
            prompt=prompt,
            num_generations=num_generations,
            input_image=original_img,  # ImageField -> R2
            input_mask=mask_img,       # ImageField -> R2
            status="PENDING",
        )

         # Avvia Celery — asincrono
        run_generation.delay(job.id)

        return JsonResponse({"ok": True, "job_id": job.id})

    # GET: mostra pagina vuota / stato iniziale
    return render(
        request,
        "image_editor/index.html",
        {
            "last_prompt": request.POST.get("prompt", "") if request.method == "POST" else "",
            "last_num_generations": request.POST.get("num_generations", "1") if request.method == "POST" else "1",
            "errors": [],
        },
    )
def job_status_view(request, job_id: int):
    """
    View per controllare lo stato di un job di generazione.

    Ritorna JSON con:
    - status: PENDING / RUNNING / COMPLETED / FAILED
    - error: eventuale messaggio di errore
    - generated_images: lista di URL (su R2) delle immagini generate
    """
    job = get_object_or_404(GenerationJob, id=job_id)
    data = {
        "job_id": job.id,
        "status": job.status,
        "error": job.error_message,
        "generated_images": job.generated_images or [],
        # volendo puoi anche inserire gli URL di input:
        # "input_image": job.input_image.url if job.input_image else None,
        # "input_mask": job.input_mask.url if job.input_mask else None,
    }
    return JsonResponse(data)

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

    # Leggo il file dallo storage (R2 o locale) tramite ImageField
    with job.output_image.open("rb") as f:
        data = f.read()

    # Ritorno i bytes come immagine PNG
    response = HttpResponse(data, content_type="image/png")

    # Evita cache aggressiva durante sviluppo
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["Pragma"] = "no-cache"

    return response
@require_POST
@csrf_protect
def remove_background(request):
    """
    Riceve un file immagine, crea un RemoveBgJob e avvia il task Celery.
    Ritorna subito job_id per polling (NON blocca).
    """
    uploaded_image = request.FILES.get("image")

    if uploaded_image is None:
        return JsonResponse({"ok": False, "error": "File 'image' mancante"}, status=400)

    # Creo job e salvo input (va su R2 se configurato)
    job = RemoveBgJob.objects.create(
        input_image=uploaded_image,
        status="PENDING",
    )

    # Avvio task asincrono
    remove_background_task.delay(job.id)

    # Risposta coerente con start_generation_view
    return JsonResponse({"ok": True, "job_id": job.id, "status": "PENDING"})


def remove_background_status_view(request, job_id: int):
    """
    Polling endpoint:
    ritorna status e, quando COMPLETED, anche image_url.
    """
    job = get_object_or_404(RemoveBgJob, id=job_id)

    image_url = None
    if job.status == "COMPLETED" and job.output_image:
        image_url = job.output_image.url

    return JsonResponse({
        "job_id": job.id,
        "status": job.status,
        "error": job.error_message,
        "image_url": image_url,
    })