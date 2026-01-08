import os
from django.shortcuts import get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.core.files import File
from django.conf import settings
from detection.models import DetectionResult
from .models import Explanation
from .utils import generate_gradcam

@login_required
def generate_explanation(request, detection_id):
    det = get_object_or_404(DetectionResult, id=detection_id, upload__user=request.user)
    if hasattr(det, "explanation"):
        return redirect("reports:detail", detection_id=det.id)

    filename = f"heatmap_{det.id}.png"
    relative = generate_gradcam(det.upload.image.path, filename)
    abs_path = os.path.join(settings.MEDIA_ROOT, relative)

    with open(abs_path, "rb") as f:
        Explanation.objects.create(
            detection=det,
            heatmap_image=File(f, name=filename)
        )

    return redirect("reports:detail", detection_id=det.id)
