from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    """Returns the Path to the examples directory at the repository root."""
    # __file__ is tests/conftest.py
    # .parent is tests/
    # .parent.parent is the repository root
    root_dir = Path(__file__).parent.parent
    return root_dir / "examples"
