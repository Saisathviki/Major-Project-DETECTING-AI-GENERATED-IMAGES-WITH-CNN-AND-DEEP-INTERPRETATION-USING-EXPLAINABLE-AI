from django.urls import path
from . import views

app_name = "source_attribution"

urlpatterns = [
    path("run/<int:detection_id>/", views.run_attribution, name="run"),
]
