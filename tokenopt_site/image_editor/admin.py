from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import GenerationJob,SuperResolutionJob,RemoveBgJob,UserUpload

admin.site.register(UserUpload)
admin.site.register(GenerationJob)
admin.site.register(SuperResolutionJob)
admin.site.register(RemoveBgJob)