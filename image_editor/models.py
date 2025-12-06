from django.db import models

# Create your models here.
from django.db import models

class GenerationJob(models.Model):
    STATUS_CHOICES = [
        ("PENDING","Pending"),
        ("RUNNING","Running"),
        ("COMPLETED","Completed"),
        ("FAILED","Failed"),
    ]

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    status=models.CharField(max_length=20,
                            choices=STATUS_CHOICES,
                            default="PENDING",
                            )
    prompt=models.TextField()
    num_generations=models.IntegerField(default=1)

    #file in input
    input_image=models.ImageField(upload_to="inputs/",)
    input_mask=models.ImageField(upload_to="masks/", null=True, blank=True)

    #path salvataggio risultati
    output_dir=models.CharField(max_length=255,null=True,blank=True)
    error_message=models.TextField(null=True,blank=True)

    def __str__(self):
        return f"Job {self.id} - {self.status}"