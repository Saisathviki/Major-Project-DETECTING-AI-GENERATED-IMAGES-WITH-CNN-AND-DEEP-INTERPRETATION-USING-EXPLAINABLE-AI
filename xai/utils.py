import os
import cv2
import numpy as np
import torch
from django.conf import settings
from detection.cnn import get_model, preprocess_image

def generate_gradcam(image_path, output_name):
    model = get_model()
    model.eval()
    # Find the last Conv2d layer in model.features for Grad-CAM
    target_layer = None
    for module in reversed(list(model.features)):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    if target_layer is None:
        # Fallback to the last feature module if no Conv2d found
        target_layer = list(model.features)[-1]

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    tensor, _ = preprocess_image(image_path)
    tensor = tensor.to(next(model.parameters()).device)

    output = model(tensor)
    target = output[0]
    model.zero_grad()
    target.backward()

    grads = gradients[0][0]   # [C,H,W]
    acts = activations[0][0]  # [C,H,W]

    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(0).cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (256, 256))

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.4 * heatmap + 0.6 * img)

    out_dir = os.path.join(settings.MEDIA_ROOT, "xai")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_name)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    fh.remove()
    bh.remove()

    return os.path.join("xai", output_name)
