from django.urls import path
from . import views

app_name = "reports"

urlpatterns = [
    path("", views.list_reports, name="list"),
    path("<int:detection_id>/", views.detail_report, name="detail"),
    path("<int:detection_id>/pdf/", views.download_pdf, name="pdf"),
]
