"""
Custom build script to compile libaccelerate_wrapper.dylib

The wrapper is loaded with ctypes, so it does not depend on the Python ABI.
The wheel is therefore tagged py3-none-macosx_<ver>_universal2: one wheel for
every Python version, installable only on macOS.
"""
import importlib.util
import platform
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel

# Load the build helper by path; importing the package would try to load the dylib.
_spec = importlib.util.spec_from_file_location(
    "_build_wrapper", Path(__file__).parent / "src" / "emerge_aasds" / "_build_wrapper.py"
)
_build_wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build_wrapper)

# pip only matches the major version for macOS 11+, so 15.5 is tagged macosx_15_0.
# The exact minimum (15.5) is enforced at import time in _interface.py.
_major = _build_wrapper.MACOSX_DEPLOYMENT_TARGET.split(".")[0]
PLATFORM_TAG = f"macosx_{_major}_0_universal2"


class BuildWrapper(build_py):
    """Compile the C wrapper alongside the Python sources."""

    def run(self):
        build_py.run(self)

        if platform.system() != 'Darwin':
            print("WARNING: emerge-aasds only works on macOS")
            print("Skipping wrapper compilation on non-macOS system")
            return

        # Editable installs import from src/, regular builds from build_lib.
        if self.editable_mode:
            output_dir = Path('src/emerge_aasds')
        else:
            output_dir = Path(self.build_lib) / 'emerge_aasds'

        print("=" * 70)
        if not _build_wrapper.build_wrapper(output_dir):
            sys.exit(1)
        print("=" * 70)


class PlatformWheel(bdist_wheel):
    """Tag the wheel as macOS-only but Python-version independent."""

    def get_tag(self):
        return "py3", "none", PLATFORM_TAG


if __name__ == '__main__':
    setup(cmdclass={'build_py': BuildWrapper, 'bdist_wheel': PlatformWheel})
