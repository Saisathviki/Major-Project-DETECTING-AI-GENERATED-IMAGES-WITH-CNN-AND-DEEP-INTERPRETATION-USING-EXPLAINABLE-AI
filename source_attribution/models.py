from django.db import models
from core.models import TimeStampedModel
from detection.models import DetectionResult

class SourceAttribution(TimeStampedModel):
    detection = models.OneToOneField(
        DetectionResult,
        on_delete=models.CASCADE,
        related_name="source_attr"
    )
    source_label = models.CharField(max_length=50)
    confidence = models.FloatField()
    notes = models.TextField(blank=True)
