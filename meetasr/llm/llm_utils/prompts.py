"""Utility for loading LLM prompts."""

import os

_PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

def load_prompts(language: str = "vi", prompt_type: str = "meeting") -> dict[str, str]:
    """Load prompt templates from the filesystem.

    Args:
        language: Language code ("vi", "en").
        prompt_type: Type of prompt (e.g. "meeting").

    Returns:
        A dictionary containing the loaded prompt strings.
        Keys: "summarize", "topics", "action_items", "decisions".
    """
    def _read(task: str) -> str:
        folder_path = os.path.join(_PROMPT_DIR, f"{task}_{language}")
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Missing prompt folder: {folder_path}")

        filename = f"{task}_{prompt_type}_{language}.txt"
        path = os.path.join(folder_path, filename)

        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()

        raise FileNotFoundError(
            f"Missing prompt file: {path} (task={task}, type={prompt_type}, lang={language})"
        )

    return {
        "summarize": _read("summarize"),
        "topics": _read("topics"),
        "action_items": _read("action_items"),
        "decisions": _read("decisions"),
    }
