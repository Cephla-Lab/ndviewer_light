"""Test-session setup shared by all tests."""

import os

# ndviewer_light is a PyQt5 app. When pytest-qt is installed alongside PyQt6 it
# imports PyQt6 first, and vispy then refuses to load its PyQt5 backend.
os.environ.setdefault("PYTEST_QT_API", "pyqt5")
