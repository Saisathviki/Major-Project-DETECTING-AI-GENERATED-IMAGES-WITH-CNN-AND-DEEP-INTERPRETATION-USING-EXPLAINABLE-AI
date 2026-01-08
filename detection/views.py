import os
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from uploads.models import UploadedImage
from .models import DetectionResult
from .cnn import predict

@login_required
def run_detection(request, upload_id):
    upload = get_object_or_404(UploadedImage, id=upload_id, user=request.user)
    if hasattr(upload, "detection"):
        return redirect("reports:detail", detection_id=upload.detection.id)

    is_real, prob, info = predict(upload.image.path)
    det = DetectionResult.objects.create(
        upload=upload,
        is_real=is_real,
        confidence=prob,
        preprocessing_info=info,
    )
    upload.processed = True
    upload.save(update_fields=["processed"])
    
    # Auto-generate heatmap and source attribution
    try:
        from xai.views import generate_explanation
        class MockRequest:
            def __init__(self, user):
                self.user = user
        generate_explanation(MockRequest(request.user), det.id)
    except Exception as e:
        messages.warning(request, f"Could not generate heatmap: {e}")
    
    try:
        from source_attribution.views import run_attribution
        run_attribution(MockRequest(request.user), det.id)
    except Exception as e:
        messages.warning(request, f"Could not generate source attribution: {e}")
    
    return redirect("reports:detail", detection_id=det.id)

@login_required
def run_batch(request):
    uploads = UploadedImage.objects.filter(user=request.user, processed=False).exclude(detection__isnull=False)
    processed_count = 0
    error_count = 0
    errors = []
    
    if not uploads.exists():
        messages.info(request, "No images to process.")
        return redirect("reports:list")
    
    for u in uploads:
        try:
            # Verify image file exists
            if not u.image or not hasattr(u.image, 'path') or not os.path.exists(u.image.path):
                error_count += 1
                errors.append(f"{u.original_filename}: File not found")
                u.processed = True
                u.save(update_fields=["processed"])
                continue
            
            is_real, prob, info = predict(u.image.path)
            det = DetectionResult.objects.create(
                upload=u,
                is_real=is_real,
                confidence=prob,
                preprocessing_info=info,
            )
            u.processed = True
            u.save(update_fields=["processed"])
            
            # Auto-generate heatmap and source attribution for batch mode
            try:
                from xai.views import generate_explanation
                class MockRequest:
                    def __init__(self, user):
                        self.user = user
                generate_explanation(MockRequest(None), det.id)
            except Exception as batch_e:
                pass  # Silent fail in batch mode to not clutter error list
            
            try:
                from source_attribution.views import run_attribution
                run_attribution(MockRequest(None), det.id)
            except Exception as batch_e:
                pass  # Silent fail in batch mode
            
            processed_count += 1
        except Exception as e:
            error_count += 1
            errors.append(f"{u.original_filename}: {str(e)}")
            u.processed = True
            u.save(update_fields=["processed"])
    
    if processed_count > 0:
        messages.success(request, f"Successfully processed {processed_count} image(s).")
    if error_count > 0:
        error_msg = f"Failed to process {error_count} image(s): " + "; ".join(errors[:3])
        if len(errors) > 3:
            error_msg += f"... and {len(errors) - 3} more"
        messages.error(request, error_msg)
    
    return redirect("reports:list")
