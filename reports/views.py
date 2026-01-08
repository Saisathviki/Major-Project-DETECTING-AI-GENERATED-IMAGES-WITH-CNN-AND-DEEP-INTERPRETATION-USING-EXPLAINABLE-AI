from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from reportlab.pdfgen import canvas
from detection.models import DetectionResult

@login_required
def list_reports(request):
    detections = DetectionResult.objects.filter(
        upload__user=request.user
    ).select_related("upload").order_by("-created_at")
    return render(request, "reports/list.html", {"detections": detections})

@login_required
def detail_report(request, detection_id):
    det = get_object_or_404(DetectionResult, id=detection_id, upload__user=request.user)
    return render(request, "reports/detail.html", {"det": det})

@login_required
def download_pdf(request, detection_id):
    det = get_object_or_404(DetectionResult, id=detection_id, upload__user=request.user)

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="report_{det.id}.pdf"'

    p = canvas.Canvas(response)
    p.setTitle("AI Image Detection Report")

    y = 800
    p.drawString(50, y, "AI Image Detection Report"); y -= 20
    p.drawString(50, y, f"Detection ID: {det.id}"); y -= 20
    p.drawString(50, y, f"Image ID: {det.upload.id}"); y -= 20
    p.drawString(50, y, f"Filename: {det.upload.original_filename}"); y -= 20
    p.drawString(50, y, f"Prediction: {'Real' if det.is_real else 'AI-generated'}"); y -= 20
    p.drawString(50, y, f"Confidence: {det.confidence:.0f}%"); y -= 20

    if hasattr(det, "source_attr"):
        p.drawString(50, y, f"Source: {det.source_attr.source_label} ({det.source_attr.confidence:.2f})"); y -= 20

    p.showPage()
    p.save()
    return response
