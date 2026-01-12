## Gemini 3 custom nodes

Two lightweight ComfyUI nodes for Gemini 3 preview models:

- **Gemini 3 Pro (Text)** – calls `gemini-3-pro-preview`, accepts optional images and media resolution.
- **Gemini 3 Pro Image** – calls `gemini-3-pro-image-preview`, lets you pick aspect ratio and resolution.
- **Gemini Seed (int32)** – normalizes any seed to a signed int32 for Gemini. Use this before the image node if your workflow produces large seeds.

Usage notes:
- Preferred: put API keys in `config.json` (ignored by git). The node input can still override.
- Fallback: set `GEMINI_API_KEY` in the environment.
- When a key hits quota/limit errors, the node automatically tries the next key.
- Image tensors are auto-converted to PNG for requests and returned as standard ComfyUI `IMAGE`.
- Image node returns both the generated image and any model text output (captions or system notes).
- Image node includes a `seed` input (`-1` = random each run). Valid range is `0` to `2147483647` (Gemini expects signed int32).
- Gemini Seed (int32) supports `random_if_negative`, `wrap`, or `clamp` modes to keep seeds in-range.

Seed mode (plain language):
- `random_if_negative`: if seed < 0, pick a fresh random seed; otherwise wrap into 0~2147483647.
- `wrap`: always wrap any number into 0~2147483647 (big numbers just loop around).
- `clamp`: numbers below 0 become 0; above 2147483647 become 2147483647.

Configuration:
- Copy `config.example.json` to `config.json` and fill in your keys.
- `config.json` sits next to `nodes.py` in the plugin folder.
- `config.json` stays local and is not committed to git.
