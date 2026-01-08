# xai/admin.py
from django.contrib import admin
from django.utils.html import format_html
from .models import Explanation

@admin.register(Explanation)
class ExplanationAdmin(admin.ModelAdmin):
    list_display = ("id", "detection", "method", "created_at", "thumb")
    search_fields = ("detection__upload__original_filename",)
    autocomplete_fields = ("detection",)

    def thumb(self, obj):
        if obj.heatmap_image:
            return format_html(
                '<img src="{}" style="height:60px;border-radius:4px;" />',
                obj.heatmap_image.url
            )
        return "-"
    thumb.short_description = "Heatmap"
