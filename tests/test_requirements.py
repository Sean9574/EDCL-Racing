import platform
from importlib.metadata import version
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def pinned():
    """{package: version} of every pinned line of requirements.txt."""
    pins = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, value = line.partition("==")
            pins[name.strip()] = value.strip()
    return pins


def test_requirements_pin_every_package_exactly():
    pins = pinned()
    assert {"numpy", "scipy", "pandas", "matplotlib", "tensorflow", "keras", "keras-tuner", "scikit-learn", "joblib",
            "pygame", "nbclient", "nbformat", "ipykernel", "jupyter_client", "pytest"} == set(pins)
    assert all(pins.values())


def test_the_environment_matches_the_pins():
    """The experiment runs in the environment requirements.txt describes (spec §15)."""
    assert platform.python_version().startswith("3.13.")
    assert {name: version(name) for name in pinned()} == pinned()
