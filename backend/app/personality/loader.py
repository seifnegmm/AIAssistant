"""Personality configuration loader."""

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_PATH = Path(__file__).parent / "default.yaml"


def load_personality(path: Path | None = None) -> dict[str, Any]:
    """Load personality configuration from YAML file.

    Args:
        path: Optional path to personality YAML file.
              Defaults to default.yaml in this directory.

    Returns:
        Dictionary with personality configuration.

    Raises:
        FileNotFoundError: If the personality file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    config_path = path or _DEFAULT_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Personality file not found: {config_path}")

    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    return config


def get_system_prompt(personality: dict[str, Any] | None = None) -> str:
    """Extract the system prompt from personality config.

    Args:
        personality: Pre-loaded personality dict. Loads default if None.

    Returns:
        The system prompt string.
    """
    if personality is None:
        personality = load_personality()

    return personality.get("system_prompt", "You are a helpful AI assistant.")
