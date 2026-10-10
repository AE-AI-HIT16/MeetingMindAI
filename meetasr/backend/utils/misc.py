"""Shared utility functions."""


def seconds_to_human(seconds: float) -> str:
    """Convert seconds to human-readable HH:MM:SS or MM:SS string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted time string, e.g. "1:23:45" or "03:12".
    """
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override values.

    Args:
        base: Base dictionary to update in-place.
        override: Dictionary with override values.

    Returns:
        Updated base dictionary.
    """
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base
