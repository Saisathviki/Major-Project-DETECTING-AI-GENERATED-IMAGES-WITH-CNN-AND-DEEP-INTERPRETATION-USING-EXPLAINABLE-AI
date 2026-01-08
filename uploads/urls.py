from django.urls import path
from . import views

app_name = "uploads"

urlpatterns = [
    path("upload/", views.upload_image, name="upload"),
    path("my/", views.my_uploads, name="my_uploads"),
]
