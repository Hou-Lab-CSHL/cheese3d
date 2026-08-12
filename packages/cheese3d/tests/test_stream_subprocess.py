import io
import sys

import pytest

from cheese3d.project import _stream_subprocess


def test_stream_subprocess_relays_stdout_to_sys_stdout(monkeypatch):
    """Output must reach whatever sys.stdout currently points to (e.g. a GUI log)."""
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    _stream_subprocess([sys.executable, "-c", "print('hello from child')"])

    assert "hello from child" in captured.getvalue()


def test_stream_subprocess_preserves_carriage_return_progress_updates(monkeypatch):
    """A bare '\\r' (tqdm-style progress redraw) must not become a new line."""
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    _stream_subprocess([
        sys.executable, "-c",
        "import sys; sys.stdout.write('1/10\\r'); sys.stdout.write('10/10\\r')",
    ])

    assert "\n" not in captured.getvalue()
    assert captured.getvalue().count("\r") == 2


def test_stream_subprocess_merges_stderr_into_the_same_stream(monkeypatch):
    """A traceback printed by the child must be visible, not silently dropped."""
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    _stream_subprocess([
        sys.executable, "-c",
        "import sys; sys.stderr.write('boom on stderr\\n')",
    ])

    assert "boom on stderr" in captured.getvalue()


def test_stream_subprocess_raises_called_process_error_on_nonzero_exit(monkeypatch):
    """A failing child must still fail the caller instead of being swallowed."""
    from subprocess import CalledProcessError

    monkeypatch.setattr(sys, "stdout", io.StringIO())

    with pytest.raises(CalledProcessError) as excinfo:
        _stream_subprocess([sys.executable, "-c", "raise SystemExit(1)"])

    assert excinfo.value.returncode == 1
