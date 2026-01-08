"""
Calibration tool for source attribution analyzer.
Tests on sample real/fake images and identifies misclassifications.

Usage:
    python calibrate.py --scan              # auto-scan real/fake folders
    python calibrate.py --scan --verbose    # detailed per-signal breakdown
    python calibrate.py label_type path     # test single image (label_type: real/fake/unknown)
"""

import sys
from pathlib import Path
from .analyzer import SourceAnalyzer

ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = ROOT / 'ml' / 'cifake' / 'real'
FAKE_DIR = ROOT / 'ml' / 'cifake' / 'fake'
UPLOAD_DIR = ROOT / 'media' / 'uploads'

EXTS = {'.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.gif'}


def test_image(img_path: Path, expected_label: str, verbose: bool = False) -> dict:
    """Test single image and return result dict."""
    if not img_path.exists():
        return {
            'path': str(img_path.relative_to(ROOT)),
            'expected': expected_label,
            'result': 'SKIP',
            'reason': 'File not found'
        }
    
    try:
        analyzer = SourceAnalyzer(str(img_path))
        if not analyzer.valid:
            return {
                'path': str(img_path.relative_to(ROOT)),
                'expected': expected_label,
                'result': 'SKIP',
                'reason': 'Could not load image'
            }
        
        report = analyzer.get_detailed_report()
        label = report['label']
        conf = report['confidence']
        
        # Map expected label to what we expect to see
        if expected_label.lower() == 'real':
            is_correct = label == 'Real camera'
        elif expected_label.lower() == 'fake':
            is_correct = label in ['GAN', 'Diffusion']
        else:
            is_correct = True
        
        result = {
            'path': str(img_path.relative_to(ROOT)),
            'expected': expected_label,
            'predicted': label,
            'confidence': conf,
            'correct': is_correct,
            'result': 'PASS' if is_correct else 'FAIL',
            'probs': report['probs'],
        }
        
        if verbose:
            result['contributions'] = {
                k: {
                    'gan': round(v['gan'], 4),
                    'real': round(v['real'], 4),
                    'diffusion': round(v['diffusion'], 4),
                }
                for k, v in report['contributions'].items()
            }
            result['raw_scores'] = {
                k: {
                    'gan': round(v.get('gan_score', 0), 4),
                    'real': round(v.get('real_score', 0), 4),
                    'diffusion': round(v.get('diffusion_score', 0), 4),
                }
                for k, v in report['scores'].items()
            }
        
        return result
    
    except Exception as e:
        return {
            'path': str(img_path.relative_to(ROOT)),
            'expected': expected_label,
            'result': 'ERROR',
            'reason': str(e)
        }


def print_result(res: dict, verbose: bool = False):
    """Print formatted result."""
    if res['result'] == 'SKIP':
        print(f"  ⊘ {res['path']}: {res['reason']}")
        return
    
    if res['result'] == 'ERROR':
        print(f"  ✗ {res['path']}: ERROR - {res['reason']}")
        return
    
    symbol = "✓" if res['result'] == 'PASS' else "✗"
    print(f"  {symbol} {res['path']}")
    print(f"      Expected: {res['expected']}, Got: {res['predicted']} (conf={res['confidence']})")
    print(f"      Probs: GAN={res['probs']['gan']:.3f}, Real={res['probs']['real']:.3f}, Diff={res['probs']['diffusion']:.3f}")
    
    if verbose and 'contributions' in res:
        print(f"      Weighted contributions:")
        for analyzer, contrib in res['contributions'].items():
            print(f"        {analyzer}: GAN={contrib['gan']:.4f}, Real={contrib['real']:.4f}, Diff={contrib['diffusion']:.4f}")


def scan_and_report(verbose: bool = False):
    """Scan real/fake folders and report statistics."""
    print("\n=== SOURCE ATTRIBUTION CALIBRATION ===\n")
    
    results = {'real': [], 'fake': []}
    
    # Test real images
    if REAL_DIR.exists():
        print(f"Testing REAL images from {REAL_DIR.relative_to(ROOT)}:")
        for img in sorted(REAL_DIR.rglob('*')):
            if img.suffix.lower() in EXTS:
                res = test_image(img, 'real', verbose)
                results['real'].append(res)
                print_result(res, verbose)
    else:
        print(f"⊘ Real images folder not found: {REAL_DIR.relative_to(ROOT)}")
    
    print()
    
    # Test fake images
    if FAKE_DIR.exists():
        print(f"Testing FAKE images from {FAKE_DIR.relative_to(ROOT)}:")
        for img in sorted(FAKE_DIR.rglob('*')):
            if img.suffix.lower() in EXTS:
                res = test_image(img, 'fake', verbose)
                results['fake'].append(res)
                print_result(res, verbose)
    else:
        print(f"⊘ Fake images folder not found: {FAKE_DIR.relative_to(ROOT)}")
    
    print()
    
    # Test uploads
    if UPLOAD_DIR.exists():
        print(f"Testing UPLOADS (unlabeled) from {UPLOAD_DIR.relative_to(ROOT)}:")
        for img in sorted(UPLOAD_DIR.rglob('*')):
            if img.suffix.lower() in EXTS:
                res = test_image(img, 'unknown', verbose)
                results['unknown'] = results.get('unknown', [])
                results['unknown'].append(res)
                print_result(res, verbose)
    
    print()
    print("=== SUMMARY ===")
    for category, tests in results.items():
        if not tests:
            continue
        passed = sum(1 for r in tests if r.get('result') == 'PASS')
        failed = sum(1 for r in tests if r.get('result') == 'FAIL')
        errors = sum(1 for r in tests if r.get('result') in ['ERROR', 'SKIP'])
        total = len(tests)
        
        if failed > 0 or errors > 0:
            print(f"{category.upper()}: {passed}/{total} correct ({failed} failed, {errors} errors)")
        else:
            print(f"{category.upper()}: {passed}/{total} correct ✓")


if __name__ == '__main__':
    verbose = '--verbose' in sys.argv
    
    if '--scan' in sys.argv:
        scan_and_report(verbose)
    elif len(sys.argv) >= 3:
        label_type = sys.argv[1]  # 'real', 'fake', or 'unknown'
        img_path = Path(sys.argv[2])
        res = test_image(img_path, label_type, verbose)
        print_result(res, verbose)
        if verbose and 'contributions' in res:
            print("\nDetailed contributions by analyzer:")
            for analyzer, contrib in res['contributions'].items():
                print(f"  {analyzer}: {contrib}")
    else:
        print(__doc__)
