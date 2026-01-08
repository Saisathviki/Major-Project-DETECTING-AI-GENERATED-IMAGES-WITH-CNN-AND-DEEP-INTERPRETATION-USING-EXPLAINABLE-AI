from django.shortcuts import get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from detection.models import DetectionResult
from .models import SourceAttribution
from .utils import simple_source_heuristic

@login_required
def run_attribution(request, detection_id):
    det = get_object_or_404(DetectionResult, id=detection_id, upload__user=request.user)
    if hasattr(det, "source_attr"):
        return redirect("reports:detail", detection_id=det.id)
    # Prefer the detection model when it's confident about real/fake
    # If detection is confident the image is real, set `Real camera` immediately.
    # Otherwise, fall back to the analyzer. If detection marked as fake but the
    # analyzer returns `Real camera`, prefer a fake label (GAN) to avoid false real positives.
    label = None
    conf = 0.0
    notes = ""

    try:
        if det.is_real and det.confidence is not None and det.confidence >= 0.65:
            label = "Real camera"
            conf = round(float(det.confidence), 2)
            notes = "Assigned from detection model (high confidence)."
        else:
            # Use analyzer to decide between GAN / Diffusion / Real
            label, conf, notes = simple_source_heuristic(det.upload.image.path)

            # If detection model flagged as fake but analyzer returned Real,
            # prefer a fake label (GAN) to avoid showing real for detected fakes.
            if hasattr(det, 'is_real') and det.is_real is False and label == "Real camera":
                label = "GAN"
                # ensure reasonable confidence
                conf = max(conf, 0.6)
                notes = "Detection model indicated fake; overriding analyzer result to GAN."
    except Exception as e:
        # Fallback: mark unknown if analyzer/detection fails
        label = "Unknown"
        conf = 0.5
        notes = f"Error during attribution: {e}"

    SourceAttribution.objects.create(
        detection=det,
        source_label=label,
        confidence=conf,
        notes=notes,
    )
    return redirect("reports:detail", detection_id=det.id)
