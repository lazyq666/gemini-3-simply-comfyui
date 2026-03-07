MISSING_DEP_LOAD_HINT = (
    "Failed to load gemini3 nodes because dependency 'google-genai' is missing. "
    "Install requirements in ComfyUI's Python environment and restart ComfyUI.\n"
    "Example: python -m pip install -r custom_nodes/gemini3/requirements.txt"
)

try:
    from .nodes import (
        Gemini3Camera3DPrompt,
        Gemini3ProPreviewText,
        GeminiSeedInt32,
        Gemini3ProImagePreview,
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
except (ModuleNotFoundError, ImportError) as exc:
    msg = str(exc).lower()
    if "google" in msg or "google-genai" in msg:
        raise ImportError(f"{MISSING_DEP_LOAD_HINT}\nOriginal error: {exc}") from exc
    raise

__all__ = [
    "Gemini3Camera3DPrompt",
    "Gemini3ProPreviewText",
    "GeminiSeedInt32",
    "Gemini3ProImagePreview",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

WEB_DIRECTORY = "./web"
