"""create_shortcut.py: the shortcut targets the environment of the Python that runs it."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def create_shortcut():
    spec = importlib.util.spec_from_file_location(
        "create_shortcut", ROOT / "create_shortcut.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_command(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = "ndviewer-light.exe" if os.name == "nt" else "ndviewer-light"
    path = directory / name
    path.write_text("")
    path.chmod(0o755)
    return path


def test_prefers_the_command_next_to_the_running_python(
    create_shortcut, tmp_path, monkeypatch
):
    venv_cmd = fake_command(tmp_path / "venv" / "Scripts")
    fake_command(tmp_path / "global")
    monkeypatch.setattr(sys, "executable", str(venv_cmd.parent / "python.exe"))
    monkeypatch.setenv("PATH", str(tmp_path / "global"))

    exe, module = create_shortcut._resolve_launch_command()

    assert Path(exe).resolve() == venv_cmd.resolve()
    assert module is None


def test_falls_back_to_path_then_to_python_module(
    create_shortcut, tmp_path, monkeypatch
):
    python = tmp_path / "venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(python))

    on_path = fake_command(tmp_path / "global")
    monkeypatch.setenv("PATH", str(on_path.parent))
    exe, module = create_shortcut._resolve_launch_command()
    assert Path(exe).resolve() == on_path.resolve() and module is None

    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert create_shortcut._resolve_launch_command() == (str(python), "ndviewer_light")
