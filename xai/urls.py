from django.urls import path
from . import views

app_name = "xai"

urlpatterns = [
    path("generate/<int:detection_id>/", views.generate_explanation, name="generate"),
]
