#!/usr/bin/env python
"""
Standalone script to build the Accelerate wrapper library.
Can be called directly (python src/emerge_aasds/_build_wrapper.py) or used by setup.py
"""

import subprocess
import sys
from pathlib import Path
import platform

# Lowest macOS that provides the Sparse API used by _wrapper.c (complex support).
# MTMetis ordering (macOS 26) is guarded at runtime in _wrapper.c.
MACOSX_DEPLOYMENT_TARGET = "15.5"
ARCHS = ("arm64", "x86_64")
DYLIB_NAME = "libaccelerate_wrapper.dylib"


def build_wrapper(output_dir=None):
    """Build libaccelerate_wrapper.dylib as a universal2 binary.

    Returns True on success, False on failure.
    """

    if platform.system() != 'Darwin':
        print("Error: emerge-aasds only works on macOS")
        return False

    here = Path(__file__).parent
    wrapper_c = here / '_wrapper.c'
    output_dylib = Path(output_dir or here) / DYLIB_NAME
    output_dylib.parent.mkdir(parents=True, exist_ok=True)

    print("Building Apple Accelerate wrapper...")

    cmd = ['clang', '-dynamiclib', '-O3']
    for arch in ARCHS:
        cmd += ['-arch', arch]
    cmd += [
        f'-mmacosx-version-min={MACOSX_DEPLOYMENT_TARGET}',
        '-install_name', f'@rpath/{DYLIB_NAME}',
        '-framework', 'Accelerate',
        '-o', str(output_dylib),
        str(wrapper_c),
    ]

    print(f"Compiling: {' '.join(cmd)}")

    try:
        subprocess.check_call(cmd)
        print(f"✓ Built: {output_dylib}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ Build failed: {e}")
        print("\nPlease ensure Xcode Command Line Tools are installed and up to date:")
        print("  xcode-select --install")
        print(f"(an SDK for macOS {MACOSX_DEPLOYMENT_TARGET} or newer is required)")
        return False

if __name__ == '__main__':
    success = build_wrapper()
    sys.exit(0 if success else 1)
