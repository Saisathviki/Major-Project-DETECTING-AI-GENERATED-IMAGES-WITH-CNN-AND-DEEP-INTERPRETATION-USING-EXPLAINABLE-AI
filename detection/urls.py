from django.urls import path
from . import views

app_name = "detection"

urlpatterns = [
    path("run/<int:upload_id>/", views.run_detection, name="run"),
    path("batch/", views.run_batch, name="batch_run"),
]
