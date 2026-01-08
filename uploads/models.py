from django.db import models
from django.contrib.auth.models import User
from core.models import TimeStampedModel

class UploadedImage(TimeStampedModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="uploads")
    image = models.ImageField(upload_to="uploads/")
    original_filename = models.CharField(max_length=255, blank=True)
    processed = models.BooleanField(default=False)

    def __str__(self):
        return self.original_filename or f"Image {self.pk}"
