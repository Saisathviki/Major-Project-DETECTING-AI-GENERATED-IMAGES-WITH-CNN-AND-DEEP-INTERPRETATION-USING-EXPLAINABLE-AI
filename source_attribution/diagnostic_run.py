"""Diagnostic runner for source attribution analyzer.

Usage:
    python diagnostic_run.py            # runs on built-in sample folders
    python diagnostic_run.py <image>    # runs on single image

This script prints label, confidence, and notes for each image it processes.
"""
import sys
import os
from pathlib import Path
from .utils import simple_source_heuristic
from .analyzer import SourceAnalyzer

# Directories to probe (relative to project root)
ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIRS = [
    ROOT / 'media' / 'uploads',
    ROOT / 'ml' / 'cifake' / 'real',
    ROOT / 'ml' / 'cifake' / 'fake',
]

EXTS = {'.jpg', '.jpeg', '.png', '.jfif', '.bmp'}


def process_image(p: Path):
    try:
        label, conf, notes = simple_source_heuristic(str(p))
    except Exception as e:
        label, conf, notes = 'Error', 0.0, f'Exception: {e}'
    print(f"{p.relative_to(ROOT)} -> {label} (conf={conf}) - {notes}")


def process_image_debug(p: Path):
    try:
        analyzer = SourceAnalyzer(str(p))
        report = analyzer.get_detailed_report()
    except Exception as e:
        print(f"{p.relative_to(ROOT)} -> Error: {e}")
        return

    print(f"{p.relative_to(ROOT)} -> {report['label']} (conf={report['confidence']}) - {report['notes']}")
    print("  Probs:", report['probs'])
    print("  Totals:", {k: round(v, 4) for k, v in report['totals'].items()})
    print("  Contributions per analyzer:")
    for k, v in report['contributions'].items():
        print(f"    {k}: gan={v['gan']:.4f}, real={v['real']:.4f}, diff={v['diffusion']:.4f}, weight={v['weight']}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img = Path(sys.argv[1])
        if img.exists():
            process_image(img)
        else:
            print(f"File not found: {img}")
        sys.exit(0)

    found = False
    for d in SAMPLE_DIRS:
        if not d.exists():
            continue
        print(f"\nScanning {d.relative_to(ROOT)}")
        for p in sorted(d.rglob('*')):
            if p.suffix.lower() in EXTS:
                found = True
                process_image(p)

    if not found:
        print("No sample images found in default directories. Provide an image path as argument.")
