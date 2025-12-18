## Gemini 3 custom nodes

Two lightweight ComfyUI nodes for Gemini 3 preview models:

- **Gemini 3 Pro (Text)** – calls `gemini-3-pro-preview`, accepts optional images and media resolution.
- **Gemini 3 Pro Image** – calls `gemini-3-pro-image-preview`, lets you pick aspect ratio and resolution.

Usage notes:
- Supply an API key in the node or set `GEMINI_API_KEY` in the environment.
- Image tensors are auto-converted to PNG for requests and returned as standard ComfyUI `IMAGE`.
- Image node returns both the generated image and any model text output (captions or system notes).
