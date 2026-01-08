"""Quick test harness for analyzer validation.

Run this to see how the analyzer performs on your sample images.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    print("Running Source Attribution Calibration...\n")
    
    # Run calibration script
    try:
        # Try with --verbose for detailed breakdowns
        result = subprocess.run(
            [sys.executable, '-m', 'source_attribution.calibrate', '--scan', '--verbose'],
            cwd=ROOT,
            capture_output=False
        )
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running calibration: {e}")
        print("\nTrying alternative method...")
        
        # Fallback: direct import
        try:
            from source_attribution.calibrate import scan_and_report
            scan_and_report(verbose=True)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            sys.exit(1)

if __name__ == '__main__':
    main()
