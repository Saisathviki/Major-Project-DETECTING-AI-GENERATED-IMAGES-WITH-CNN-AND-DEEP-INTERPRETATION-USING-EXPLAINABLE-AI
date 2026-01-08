from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from uploads.models import UploadedImage
from detection.models import DetectionResult

@login_required
def home(request):
    total_uploads = UploadedImage.objects.filter(user=request.user).count()
    total_detections = DetectionResult.objects.filter(upload__user=request.user).count()
    latest = DetectionResult.objects.filter(
        upload__user=request.user
    ).order_by("-created_at")[:5]

    return render(request, "dashboard/home.html", {
        "total_uploads": total_uploads,
        "total_detections": total_detections,
        "latest": latest,
    })
