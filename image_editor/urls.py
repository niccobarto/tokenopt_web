from django.urls import path
from . import views

urlpatterns = [
    path('', views.start_generation_view, name='start_generation'),
    path("job-status/<int:job_id>/", views.job_status_view, name="job_status"),
]