from django.db import models
from core.models import TimeStampedModel
from detection.models import DetectionResult

class Explanation(TimeStampedModel):
    detection = models.OneToOneField(
        DetectionResult,
        on_delete=models.CASCADE,
        related_name="explanation"
    )
    heatmap_image = models.ImageField(upload_to="xai/")
    method = models.CharField(max_length=50, default="GradCAM")
