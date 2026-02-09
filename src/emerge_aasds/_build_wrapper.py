#!/usr/bin/env python
"""
Standalone script to build the Accelerate wrapper library.
Can be called directly or used by setup.py
"""

import subprocess
import sys
from pathlib import Path
import platform

def build_wrapper():
    """Build the libaccelerate_wrapper.dylib"""
    
    if platform.system() != 'Darwin':
        print("Error: emerge-aasds only works on macOS")
        return False
    
    # Get paths relative to this file
    here = Path(__file__).parent
    wrapper_c = here / '_wrapper.c'
    output_dylib = here / 'libaccelerate_wrapper.dylib'
    
    print("Building Apple Accelerate wrapper...")
    
    cmd = [
        'clang',
        '-dynamiclib',
        '-O3',
        '-framework', 'Accelerate',
        '-o', str(output_dylib),
        str(wrapper_c),
    ]
    
    try:
        subprocess.check_call(cmd)
        print(f"✓ Built: {output_dylib}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ Build failed: {e}")
        print("\nPlease install Xcode Command Line Tools:")
        print("  xcode-select --install")
        return False

if __name__ == '__main__':
    success = build_wrapper()
    sys.exit(0 if success else 1)