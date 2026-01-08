# core/models.py (add if you want)
from django.db import models
from .models import TimeStampedModel

class ModelInfo(TimeStampedModel):
    name = models.CharField(max_length=100, default="CNN CIFAKE")
    version = models.CharField(max_length=50, default="v1")
    file_path = models.CharField(max_length=255)
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.name} ({self.version})"
