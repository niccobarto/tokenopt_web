from django.db import models
import uuid
# Create your models here.
from django.db import models
from django.db.models import JSONField
from tokenopt_site.settings import TTO_JOBS_ROOT_RELATIVE,REMOVEBG_ROOT_RELATIVE,SUPERRES_ROOT_RELATIVE

STATUS_CHOICES = [
        ("PENDING","Pending"),
        ("RUNNING","Running"),
        ("COMPLETED","Completed"),
        ("FAILED","Failed"),
    ]

def generation_input_path(instance, filename):
    return f"{TTO_JOBS_ROOT_RELATIVE}/job_{instance.id}/inputs/{filename}"
def background_input_path(instance, filename):
    return f"{REMOVEBG_ROOT_RELATIVE}/job_{instance.id}/inputs/{filename}"
def superres_input_path(instance, filename):
    return f"{SUPERRES_ROOT_RELATIVE}/job_{instance.id}/inputs/{filename}"

class UserUpload(models.Model):
    """
    Rappresenta l’immagine originale caricata dall’utente (con eventuale sfondo)
    e fa da "contenitore" per tutti i job derivati (removebg, superres, generation).
    """
    created_at = models.DateTimeField(auto_now_add=True)

    # L’immagine originale (con sfondo) a cui vuoi sempre poter tornare
    original_image = models.ImageField(upload_to="uploads/originals/")

    def __str__(self):
        return f"UserUpload {self.id}"


class GenerationJob(models.Model):

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status=models.CharField(max_length=20,
                            choices=STATUS_CHOICES,
                            default="PENDING",
                            )
    prompt=models.TextField()
    num_generations=models.IntegerField(default=1)

    #file in input
    input_image=models.ImageField(upload_to=generation_input_path,null=True,blank=True)
    input_mask=models.ImageField(upload_to=generation_input_path, null=True, blank=True)

    generated_images = models.JSONField(default=list, blank=True)

    #path salvataggio risultati
    error_message=models.TextField(null=True,blank=True)

    upload=models.ForeignKey(UserUpload,on_delete=models.CASCADE,related_name="generation_jobs",null=True,blank=True)

    def __str__(self):
        return f"Job {self.id} - {self.status}"



class SuperResolutionJob(models.Model):
    created_at=models.DateTimeField(auto_now_add=True)
    status=models.CharField(max_length=20,default="PENDING",choices=STATUS_CHOICES)
    input_image=models.ImageField(upload_to=superres_input_path,)
    superres_image=models.JSONField(default=list,blank=True)

    error_message=models.TextField(null=True,blank=True)

    upload = models.ForeignKey(
        UserUpload,
        on_delete=models.CASCADE,
        related_name="superres_jobs",
        null=True,
        blank=True
    )

class RemoveBgJob(models.Model):
    """
    Job dedicato alla rimozione dello sfondo.
    Usato per tracciare stato, input e output (singolo).
    """

    ALLOWED_MODELS=[("u2net","U2NET"),("sam","SAM"),("isnet-general-use","ISNET"),("birefnet-general","BIREFNET")]

    # Stato del job (come GenerationJob)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="PENDING")
    model_selected = models.CharField(max_length=20, choices=ALLOWED_MODELS, default="u2net")
    # Messaggio di errore eventuale
    error_message = models.TextField(blank=True, null=True)

    # Input e output (ImageField -> va su R2 se DEFAULT_FILE_STORAGE punta a R2)
    input_image = models.ImageField(upload_to=background_input_path)
    output_image = models.JSONField(default=list, blank=True)

    # Timestamps utili (opzionali ma consigliati)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    upload = models.ForeignKey(
        UserUpload,
        on_delete=models.CASCADE,
        related_name="removebg_jobs",
        null=True,
        blank=True
    )
    def __str__(self):
        return f"RemoveBgJob(id={self.id}, status={self.status})"