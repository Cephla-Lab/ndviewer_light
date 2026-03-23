#!/usr/bin/env python3
"""Create a desktop shortcut for NDViewer Light."""

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "NDViewer Light"
COMMAND_NAME = "ndviewer-light"
MODULE_NAME = "ndviewer_light"
BUNDLE_ID = "com.cephla.ndviewer-light"


def _get_version():
    try:
        return importlib.metadata.version(MODULE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _resolve_launch_command():
    """Return (exe_path, None) if installed, else (python_path, module_name)."""
    exe = shutil.which(COMMAND_NAME)
    if exe:
        return exe, None
    return sys.executable, MODULE_NAME


def create_macos_app(exe, module):
    app_dir = Path.home() / "Applications" / f"{APP_NAME}.app"
    contents = app_dir / "Contents"
    macos = contents / "MacOS"

    macos.mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(parents=True, exist_ok=True)

    if module:
        launch_cmd = f'exec "{exe}" -m {module} "$@"'
    else:
        launch_cmd = f'exec "{exe}" "$@"'

    launcher = macos / COMMAND_NAME
    launcher.write_text(f"#!/bin/bash\n{launch_cmd}\n")
    os.chmod(launcher, 0o755)

    version = _get_version()
    plist = contents / "Info.plist"
    plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>{APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>{APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>{BUNDLE_ID}</string>
    <key>CFBundleVersion</key>
    <string>{version}</string>
    <key>CFBundleExecutable</key>
    <string>{COMMAND_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
</dict>
</plist>
""")

    print(f"Created: {app_dir}")
    print("You can drag it to the Dock from ~/Applications.")


def create_windows_shortcut(exe, module):
    target = exe
    args = f"-m {module}" if module else ""

    desktop = Path.home() / "Desktop"
    shortcut_path = desktop / f"{APP_NAME}.lnk"

    ps_script = f"""
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut("{shortcut_path}")
$sc.TargetPath = "{target}"
$sc.Arguments = '{args}'
$sc.WorkingDirectory = "{Path.home()}"
$sc.Description = "{APP_NAME} - 5D Image Viewer"
$sc.Save()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        check=True,
    )
    print(f"Created: {shortcut_path}")


def main():
    exe, module = _resolve_launch_command()
    system = platform.system()
    if system == "Darwin":
        create_macos_app(exe, module)
    elif system == "Windows":
        create_windows_shortcut(exe, module)
    else:
        print(f"Desktop shortcut creation is not supported on {system}.")
        print(f"You can run the viewer with: {COMMAND_NAME}")
        sys.exit(1)


if __name__ == "__main__":
    main()
