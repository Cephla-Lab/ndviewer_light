"""Entry point for running ndviewer_light as a module: python -m ndviewer_light"""

import sys

from .core import main

path = sys.argv[1] if len(sys.argv) > 1 else None
main(path)
