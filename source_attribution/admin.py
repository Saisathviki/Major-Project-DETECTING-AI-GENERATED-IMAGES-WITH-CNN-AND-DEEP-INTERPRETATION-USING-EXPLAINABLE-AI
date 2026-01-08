# source_attribution/admin.py
from django.contrib import admin
from .models import SourceAttribution

@admin.register(SourceAttribution)
class SourceAttributionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "detection",
        "source_label",
        "confidence",
        "created_at",
    )
    list_filter = ("source_label", "created_at")
    search_fields = (
        "source_label",
        "detection__upload__original_filename",
        "detection__upload__user__username",
    )
    autocomplete_fields = ("detection",)
