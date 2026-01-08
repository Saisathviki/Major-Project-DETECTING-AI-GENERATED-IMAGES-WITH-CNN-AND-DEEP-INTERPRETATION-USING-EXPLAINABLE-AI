from django import forms
from .models import UploadedImage

IMAGE_TYPES = ["image/jpeg", "image/png", "image/bmp"]


class SingleUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ["image"]

    def clean_image(self):
        img = self.cleaned_data["image"]
        if img.content_type not in IMAGE_TYPES:
            raise forms.ValidationError("Only JPEG, PNG, BMP are allowed.")
        return img


# Custom widget which allows multiple files
class MultiFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultiUploadForm(forms.Form):
    images = forms.FileField(
        widget=MultiFileInput(attrs={
            "multiple": True,
            "class": "form-control",
        })
    )

    def clean(self):
        cleaned_data = super().clean()
        # Don't validate in clean() when using prefixes, let validation happen in view
        return cleaned_data
    
    def get_files(self, files_dict):
        """Get files from request.FILES with proper prefix handling"""
        # When prefix is used, the field name becomes prefix-fieldname
        prefix = self.prefix if self.prefix else ""
        field_name = f"{prefix}-images" if prefix else "images"
        return files_dict.getlist(field_name)
