from __future__ import annotations

from importlib.resources import files


def example_path(name: str) -> str:
    """Return an absolute path to a packaged example structure file."""
    return str(files("findspingroup.examples").joinpath(name))


__all__ = ["example_path"]
