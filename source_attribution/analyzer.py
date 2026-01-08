"""
Multi-factor source attribution analyzer.
Combines multiple heuristics to classify image source: Real Camera, GAN, or Diffusion.
"""

import cv2
import numpy as np
from typing import Tuple


class SourceAnalyzer:
    """Analyzes image characteristics to determine likely source."""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.img_rgb = cv2.imread(image_path)
        self.img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.valid = self.img_rgb is not None and self.img_gray is not None
        
        if self.valid:
            self.img_rgb = cv2.resize(self.img_rgb, (256, 256))
            self.img_gray = cv2.resize(self.img_gray, (256, 256))
    
    def analyze(self) -> Tuple[str, float, str]:
        """
        Analyze image and return (label, confidence, notes).
        
        Returns:
            - label: "Real camera", "GAN", or "Diffusion"
            - confidence: 0.0-0.99 float
            - notes: explanation string
        """
        if not self.valid:
            return "Unknown", 0.5, "Could not read image file."
        
        scores = {}
        
        # Get individual scores from different analyzers
        scores['fft'] = self._analyze_fft()
        scores['color'] = self._analyze_color()
        scores['texture'] = self._analyze_texture()
        scores['noise'] = self._analyze_noise()
        
        # Aggregate scores
        result_label, result_conf, result_notes = self._aggregate_scores(scores)
        
        return result_label, result_conf, result_notes
    
    def _analyze_fft(self) -> dict:
        """Frequency domain analysis."""
        f = np.fft.fft2(self.img_gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Center energy (innermost region)
        r_center = min(h, w) // 8
        center_energy = magnitude[cy-r_center:cy+r_center, cx-r_center:cx+r_center].sum()
        
        # Edge energy (outer band)
        r_inner = min(h, w) // 4
        r_outer = min(h, w) // 2
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        outer_mask = (dist >= r_inner) & (dist <= r_outer)
        edge_energy = magnitude[outer_mask].sum() if outer_mask.sum() > 0 else 1.0
        
        ratio = center_energy / (edge_energy + 1e-6)
        
        # Classification based on energy distribution
        # Adjusted thresholds to reduce GAN false positives:
        # GANs: concentrated center (ratio > 2.0)
        # Real: balanced distribution (0.9-1.6)
        # Diffusion: more edge energy (ratio < 0.75)
        
        # Map ratio into soft scores
        gan_score = 0.0
        if ratio > 2.0:
            gan_score = min((ratio - 2.0) / 2.5, 1.0)

        real_score = max(0.0, 1.0 - abs(ratio - 1.2) / 1.4)

        diffusion_score = 0.0
        if ratio < 0.75:
            diffusion_score = min((0.75 - ratio) / 0.75, 1.0)

        result = {
            'ratio': ratio,
            'gan_score': gan_score,
            'real_score': real_score,
            'diffusion_score': diffusion_score,
        }
        return result
    
    def _analyze_color(self) -> dict:
        """Color space analysis."""
        hsv = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2HSV)
        
        # Calculate saturation statistics
        saturation = hsv[:, :, 1].astype(float) / 255.0
        sat_mean = saturation.mean()
        sat_std = saturation.std()
        
        # GANs tend to have more uniform saturation
        # Real images have more natural variation
        # Diffusion models also tend toward uniformity
        
        uniformity = 1.0 - sat_std  # 0-1, higher = more uniform
        
        result = {
            'sat_mean': sat_mean,
            'sat_std': sat_std,
            'uniformity': uniformity,
            'gan_score': uniformity * 0.3,  # GANs slightly favor uniform saturation
            'real_score': sat_std * 0.5,    # Real images have natural variation
            'diffusion_score': uniformity * 0.2,
        }
        return result
    
    def _analyze_texture(self) -> dict:
        """Texture analysis using Laplacian variance."""
        laplacian = cv2.Laplacian(self.img_gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Calculate local patterns
        # Real images: natural texture variation
        # GANs: sometimes over-smooth or has repetitive patterns
        # Diffusion: usually smooth
        
        # Normalize variance
        normalized_var = min(laplacian_var / 1000.0, 1.0)
        
        result = {
            'laplacian_var': laplacian_var,
            'normalized_var': normalized_var,
            'gan_score': 0.0,
            'real_score': normalized_var * 0.6,  # increase weight for texture
            'diffusion_score': (1.0 - normalized_var) * 0.35,
        }
        return result
    
    def _analyze_noise(self) -> dict:
        """Noise analysis."""
        # Calculate gradient magnitude to detect noise patterns
        grad_x = cv2.Sobel(self.img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.img_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # High-frequency noise detection
        noise_level = gradient_mag.std() / (gradient_mag.mean() + 1e-6)
        normalized_noise = min(noise_level / 2.0, 1.0)
        
        # Real camera images have camera sensor noise
        # GANs typically lack natural noise
        # Diffusion models may have less consistent noise
        
        result = {
            'noise_level': normalized_noise,
            'gan_score': (1.0 - normalized_noise) * 0.25,
            'real_score': normalized_noise * 0.6,  # increase noise weight for real images
            'diffusion_score': 0.0,
        }
        return result
    
    def _aggregate_scores(self, scores: dict) -> Tuple[str, float, str]:
        """Aggregate individual scores into final classification."""
        # Weighted aggregation: give more weight to noise and texture (strong real indicators)
        weights = {
            'fft': 0.4,
            'color': 0.15,
            'texture': 0.25,
            'noise': 0.2,
        }

        gan_total = 0.0
        real_total = 0.0
        diffusion_total = 0.0
        for key, s in scores.items():
            w = weights.get(key, 0.2)
            gan_total += s.get('gan_score', 0) * w
            real_total += s.get('real_score', 0) * w
            diffusion_total += s.get('diffusion_score', 0) * w
        
        # Normalize
        total = gan_total + real_total + diffusion_total + 1e-8
        gan_prob = gan_total / total
        real_prob = real_total / total
        diffusion_prob = diffusion_total / total
        
        # Improved decision logic with higher thresholds and margin-based confidence
        # Use max margin between top two candidates as confidence indicator
        probs_sorted = sorted([gan_prob, real_prob, diffusion_prob], reverse=True)
        margin = probs_sorted[0] - probs_sorted[1]
        
        # Classification thresholds: require stronger signal for each category
        # Real camera is "default" if nothing else is strong enough
        gan_threshold = 0.38  # GAN needs stronger signal
        diffusion_threshold = 0.35  # Diffusion needs strong signal
        real_threshold = 0.0   # Real is default if others don't qualify

        if gan_prob >= gan_threshold and gan_prob > real_prob and gan_prob > diffusion_prob:
            label = "GAN"
            # GAN confidence scales with margin and absolute probability
            conf = 0.55 + (gan_prob * 0.30 + margin * 0.10)
            notes = "Image characteristics indicate GAN-generated source (concentrated frequency energy)."
        elif diffusion_prob >= diffusion_threshold and diffusion_prob > real_prob and diffusion_prob > gan_prob:
            label = "Diffusion"
            conf = 0.55 + (diffusion_prob * 0.30 + margin * 0.10)
            notes = "Image characteristics suggest diffusion model generation (uniform spectrum)."
        else:
            label = "Real camera"
            # Real confidence: higher when real_prob is strong OR when neither GAN nor diffusion qualify
            if real_prob > 0.3:
                conf = 0.55 + (real_prob * 0.30 + margin * 0.10)
            else:
                # Weak real signal but others didn't qualify either
                conf = 0.50 + (1.0 - max(gan_prob, diffusion_prob)) * 0.25
            notes = "Image characteristics consistent with camera-captured image."
        
        # Clamp confidence
        conf = max(0.5, min(conf, 0.99))
        conf = round(conf, 2)
        
        return label, conf, notes

    def get_detailed_report(self, scores: dict = None) -> dict:
        """Return a detailed breakdown of per-analyzer scores and final probabilities.

        If `scores` is not provided, it will compute analyzer scores.
        """
        if scores is None:
            scores = {
                'fft': self._analyze_fft(),
                'color': self._analyze_color(),
                'texture': self._analyze_texture(),
                'noise': self._analyze_noise(),
            }

        weights = {
            'fft': 0.4,
            'color': 0.15,
            'texture': 0.25,
            'noise': 0.2,
        }

        contributions = {}
        gan_total = real_total = diffusion_total = 0.0
        for key, s in scores.items():
            w = weights.get(key, 0.2)
            g = s.get('gan_score', 0) * w
            r = s.get('real_score', 0) * w
            d = s.get('diffusion_score', 0) * w
            contributions[key] = {'gan': g, 'real': r, 'diffusion': d, 'weight': w}
            gan_total += g
            real_total += r
            diffusion_total += d

        total = gan_total + real_total + diffusion_total + 1e-8
        gan_prob = gan_total / total
        real_prob = real_total / total
        diffusion_prob = diffusion_total / total

        # Determine label and confidence (reuse aggregation logic)
        if gan_prob >= real_prob and gan_prob >= diffusion_prob and gan_prob > 0.35:
            label = "GAN"
            conf = 0.55 + gan_prob * 0.35
            notes = "Image characteristics suggest GAN-generated source."
        elif diffusion_prob >= gan_prob and diffusion_prob >= real_prob and diffusion_prob > 0.35:
            label = "Diffusion"
            conf = 0.55 + diffusion_prob * 0.35
            notes = "Image characteristics suggest diffusion model generation."
        else:
            label = "Real camera"
            conf = 0.55 + real_prob * 0.35
            notes = "Image characteristics consistent with camera capture."

        conf = max(0.5, min(conf, 0.99))
        conf = round(conf, 2)

        return {
            'scores': scores,
            'contributions': contributions,
            'totals': {'gan': gan_total, 'real': real_total, 'diffusion': diffusion_total},
            'probs': {'gan': round(float(gan_prob), 3), 'real': round(float(real_prob), 3), 'diffusion': round(float(diffusion_prob), 3)},
            'label': label,
            'confidence': conf,
            'notes': notes,
        }


def simple_source_heuristic(image_path: str) -> Tuple[str, float, str]:
    """
    Determine image source (Real camera, GAN, or Diffusion).
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (label, confidence, notes)
    """
    analyzer = SourceAnalyzer(image_path)
    return analyzer.analyze()
