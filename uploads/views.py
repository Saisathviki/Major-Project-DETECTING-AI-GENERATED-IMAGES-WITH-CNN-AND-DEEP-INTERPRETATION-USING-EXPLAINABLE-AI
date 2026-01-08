from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import SingleUploadForm, MultiUploadForm
from .models import UploadedImage

@login_required
def upload_image(request):
    if request.method == "POST":
        single_form = SingleUploadForm(request.POST, request.FILES, prefix="single")
        multi_form = MultiUploadForm(request.POST, request.FILES, prefix="multi")

        if "single-submit" in request.POST:
            if single_form.is_valid():
                obj = single_form.save(commit=False)
                obj.user = request.user
                obj.original_filename = obj.image.name
                obj.save()
                messages.success(request, "Image uploaded.")
                return redirect("detection:run", upload_id=obj.id)
            else:
                for error in single_form.non_field_errors():
                    messages.error(request, error)

        if "multi-submit" in request.POST:
            # Get files with prefix handling
            files = multi_form.get_files(request.FILES)
            
            if not files:
                messages.error(request, "Please select at least one image to upload.")
            else:
                # Validate file types
                valid_files = []
                for f in files:
                    if f.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
                        messages.error(request, f"Invalid file type for {f.name}. Only JPEG, PNG, BMP allowed.")
                    else:
                        valid_files.append(f)
                
                if valid_files:
                    for f in valid_files:
                        obj = UploadedImage(
                            user=request.user,
                            image=f,
                            original_filename=f.name,
                        )
                        obj.save()
                    messages.success(request, f"{len(valid_files)} image(s) uploaded. Processing...")
                    return redirect("detection:batch_run")
    else:
        single_form = SingleUploadForm(prefix="single")
        multi_form = MultiUploadForm(prefix="multi")

    return render(request, "uploads/upload.html", {
        "single_form": single_form,
        "multi_form": multi_form,
    })

@login_required
def my_uploads(request):
    images = UploadedImage.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "uploads/my_uploads.html", {"images": images})
