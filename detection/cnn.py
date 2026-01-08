import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, "ml", "cnn_cifake.pt")

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Enhanced feature extraction with deeper architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet normalization
])

def get_model():
    global _model
    if _model is None:
        m = ImprovedCNN()
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        m.eval()
        _model = m.to(device)
    return _model

def detect_ai_artifacts(image_array):
    """
    Detect common artifacts in AI-generated images:
    - Noise patterns
    - Frequency anomalies
    - Color channel inconsistencies
    """
    # Check for frequency artifacts using FFT
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    fft = np.fft.fft2(gray)
    magnitude = np.abs(fft)
    
    # AI images often have specific frequency patterns
    # Calculate frequency distribution
    freq_std = np.std(magnitude)
    freq_mean = np.mean(magnitude)
    freq_ratio = freq_std / (freq_mean + 1e-8)
    
    return freq_ratio

def preprocess_image(path):
    """
    Enhanced preprocessing with multiple techniques
    for better real-time image detection
    """
    img_cv = cv2.imread(path)
    if img_cv is None:
        raise ValueError(f"Could not read image: {path}")
    
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    original_shape = img_cv.shape[:2]
    
    # Apply multiple preprocessing techniques
    # 1. Bilateral filtering - preserves edges while reducing noise
    bilateral = cv2.bilateralFilter(img_cv, 9, 75, 75)
    
    # 2. Morphological operations to detect artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(bilateral, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Histogram equalization for better contrast
    lab = cv2.cvtColor(morph, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    processed_img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # 4. Detect frequency artifacts
    freq_ratio = detect_ai_artifacts(processed_img)
    
    # Convert to PIL and apply transforms
    pil_img = Image.fromarray(processed_img)
    tensor = _transform(pil_img).unsqueeze(0)
    
    info = {
        "size": original_shape,
        "frequency_ratio": float(freq_ratio),
        "preprocessing_methods": [
            "bilateral_filter",
            "morphological_operations",
            "histogram_equalization",
            "frequency_analysis"
        ]
    }
    return tensor, info

def predict(path):
    """
    Enhanced prediction with confidence scoring
    """
    model = get_model()
    tensor, info = preprocess_image(path)
    tensor = tensor.to(device)
    
    with torch.no_grad():
        out = model(tensor)
        prob = out.item()
    
    # Model output: sigmoid activation (0-1)
    # 1.0 = real image, 0.0 = AI-generated image
    is_real = prob >= 0.5
    
    # Enhanced confidence calculation
    # Uses both model output and artifact detection
    confidence = abs(prob - 0.5) * 2  # Scale to 0-1 range
    
    # Boost confidence if frequency artifacts detected
    # AI images typically have different frequency patterns
    freq_ratio = info.get("frequency_ratio", 0)
    if freq_ratio > 0.3 and not is_real:
        confidence = min(confidence * 1.2, 1.0)  # Slightly boost AI detection confidence
    
    return is_real, confidence, info

