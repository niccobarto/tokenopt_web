from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('start-generation/', views.start_generation_view, name='start_generation'),
    path("create-upload/", views.create_upload, name="create_upload"),
    path("job-status/<int:job_id>/", views.generation_status_view, name="job_status"),
    path("remove-background/", views.remove_background, name="remove_background"),
    path("remove-background-status/<int:job_id>/", views.remove_background_status_view,
         name="remove_background_status"),
    path("remove-background-result/<int:job_id>/", views.remove_background_result_view,
         name="remove_background_result",
         ),
    path("superres-start/", views.start_superres, name="superres_start"),
    path("superres-status/<int:job_id>/", views.superres_status, name="superres_status"),

]
