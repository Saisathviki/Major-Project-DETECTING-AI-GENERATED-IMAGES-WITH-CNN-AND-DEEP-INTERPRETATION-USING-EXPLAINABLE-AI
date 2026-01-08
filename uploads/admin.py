# uploads/admin.py
from django.contrib import admin
from django.utils.html import format_html
from .models import UploadedImage

@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ("id", "thumb", "user", "original_filename", "processed", "created_at")
    list_filter = ("processed", "created_at")
    search_fields = ("original_filename", "user__username")
    readonly_fields = ("preview", "created_at", "updated_at")

    def thumb(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="height:50px;border-radius:4px;" />',
                obj.image.url
            )
        return "-"
    thumb.short_description = "Preview"

    def preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height:300px;border-radius:6px;" />',
                obj.image.url
            )
        return "-"
