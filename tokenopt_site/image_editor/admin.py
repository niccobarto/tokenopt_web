from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import GenerationJob,SuperResolutionJob,RemoveBgJob

admin.site.register(GenerationJob)
admin.site.register(SuperResolutionJob)
admin.site.register(RemoveBgJob)