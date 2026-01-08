# detection/admin.py
from django.contrib import admin
from .models import DetectionResult

@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "upload_id",
        "get_user",
        "is_real",
        "confidence",
        "created_at",
    )
    list_filter = ("is_real", "created_at")
    search_fields = ("upload__original_filename", "upload__user__username")
    autocomplete_fields = ("upload",)

    def get_user(self, obj):
        return obj.upload.user.username
    get_user.short_description = "User"
