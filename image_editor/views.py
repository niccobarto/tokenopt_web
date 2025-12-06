# image_editor/views.py
import base64
import os
from io import BytesIO
from django.shortcuts import render,redirect,get_object_or_404
from django.http import HttpRequest, HttpResponse, JsonResponse
from PIL import Image
from .models import GenerationJob
from .tasks import run_generation
from django.conf import settings
from django.core.files.base import ContentFile


# Create your views here.

def dataurl_to_contentfile(dataurl: str, filename: str) -> ContentFile | None:
    """
    Converte una stringa dataURL (es. 'data:image/png;base64,...')
    in un ContentFile Django con un nome di file.
    """
    if not dataurl:
        return None

    try:
        header, b64data = dataurl.split(",", 1)
    except ValueError:
        return None

    # header esempio: "data:image/png;base64"
    # estensione = "png"
    mime_part = header.split(";")[0]          # "data:image/png"
    ext = mime_part.split("/")[-1]           # "png"

    try:
        img_bytes = base64.b64decode(b64data)
    except base64.binascii.Error:
        return None

    return ContentFile(img_bytes, name=f"{filename}.{ext}")

def start_generation_view(request):
    """View principale
    -GET: mostra la pagina dell'editor di maschera
    -POST: riceve gli input (prompt,immagine, maschera) e genera le immagini con TTO
    """
    if request.method=="POST":
        prompt=request.POST.get("prompt","").strip()
        num_generations=request.POST.get("num_generations","1").strip()
        original_dataurl=request.POST.get("original_image","").strip()
        mask_dataurl=request.POST.get("mask_image","").strip()

        #Validazione lato server parallelo a JS
        errors=[]
        if not prompt:
            errors.append("Il prompt non può essere vuoto.")
        try:
            num_generations=int(num_generations)
        except ValueError:
            num_generations=1
            errors.append("Numero di generazioni non valido, uso 1.")
        if num_generations<1 or num_generations>4:
            num_generations=1
            errors.append("Numero di generazioni deve essere tra 1 e 4, uso 1.")

        #Decodifica delle immagini
        original_img=dataurl_to_contentfile(original_dataurl,"original")
        mask_img=dataurl_to_contentfile(mask_dataurl,"mask")

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

        job=GenerationJob.objects.create(
            prompt=prompt,
            num_generations=num_generations,
            input_image=original_img,
            input_mask=mask_img,
            status="PENDING",
        )

        #Lancio task celery
        run_generation.delay(job.id)
        return JsonResponse({"ok":True, "job_id": job.id})

    return render(request,
                  "image_editor/index.html",
                  {
                      "last_prompt": request.POST.get("prompt", "") if request.method == "POST" else "",
                      "last_num_generations": request.POST.get("num_generations","1") if request.method == "POST" else "1",
                      "errors": [] ,
                  })

def job_status_view(request, job_id: int):
    """View per controllare lo stato di un job di generazione
    Ritorna JSON con lo stato e le immagini generate (se pronte)
    """
    job = get_object_or_404(GenerationJob, id=job_id)
    data={
        "job_id": job.id,
        "status": job.status,
        "error": job.error_message,
        "images":[],
    }

    if job.status == "COMPLETED" and job.output_dir:
        rel_dir=os.path.relpath(job.output_dir, settings.MEDIA_ROOT)
        for fname in os.listdir(job.output_dir):
            if fname.lower().endswith((".png",".jpg",".jpeg",".webp")):
                url=settings.MEDIA_URL+f"{rel_dir}/{fname}"
                data["images"].append(url)
    return JsonResponse(data)