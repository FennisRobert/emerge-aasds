"""
Custom build script to compile accelerate_wrapper.dylib
"""
import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py
import platform

class BuildWrapper(build_py):
    """Custom build command to compile the C wrapper"""
    
    def run(self):
        # Only build on macOS
        if platform.system() != 'Darwin':
            print("WARNING: emerge-aasds only works on macOS")
            print("Skipping wrapper compilation on non-macOS system")
            build_py.run(self)
            return
        
        # Get paths
        src_dir = Path('src/emerge_aasds')
        wrapper_c = src_dir / '_wrapper.c'
        wrapper_h = src_dir / '_wrapper.h'
        output_dylib = src_dir / 'libaccelerate_wrapper.dylib'
        
        print("="*70)
        print("Building Apple Accelerate wrapper library...")
        print("="*70)
        
        if not wrapper_c.exists():
            raise FileNotFoundError(f"Source file not found: {wrapper_c}")
        
        # Compile command
        cmd = [
            'clang',
            '-dynamiclib',
            '-O3',
            '-framework', 'Accelerate',
            '-o', str(output_dylib),
            str(wrapper_c),
        ]
        
        print(f"Compiling: {' '.join(cmd)}")
        
        try:
            subprocess.check_call(cmd)
            print(f"✓ Built: {output_dylib}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to compile wrapper: {e}")
            print("\nPlease ensure Xcode Command Line Tools are installed:")
            print("  xcode-select --install")
            sys.exit(1)
        except FileNotFoundError:
            print("✗ clang not found")
            print("\nPlease install Xcode Command Line Tools:")
            print("  xcode-select --install")
            sys.exit(1)
        
        print("="*70)
        
        # Continue with normal build
        build_py.run(self)

if __name__ == '__main__':
    setup(cmdclass={'build_py': BuildWrapper})