from django.db import models
from core.models import TimeStampedModel
from uploads.models import UploadedImage

class DetectionResult(TimeStampedModel):
    upload = models.OneToOneField(
        UploadedImage,
        on_delete=models.CASCADE,
        related_name="detection"
    )
    is_real = models.BooleanField()
    confidence = models.FloatField()
    preprocessing_info = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Detection #{self.pk} for upload {self.upload_id}"
