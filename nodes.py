import base64
import io
import math
import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from google import genai
from google.genai import types


def _require_api_key(user_key: str) -> str:
    key = (user_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key:
        raise ValueError("API key is required. Provide it in the node or set GEMINI_API_KEY.")
    return key


def _tensor_to_part(image_tensor: torch.Tensor, media_resolution: Optional[str]) -> types.Part:
    array = np.clip(image_tensor[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    blob = types.Blob(mime_type="image/png", data=buffer.getvalue())
    kwargs = {"inline_data": blob}
    if media_resolution:
        kwargs["media_resolution"] = {"level": media_resolution}
    return types.Part(**kwargs)


def _bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    if isinstance(image_bytes, str):
        image_bytes = base64.b64decode(image_bytes)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    array = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None, ...]


def _gather_text_from_parts(parts: List[types.Part]) -> str:
    collected = []
    for part in parts:
        if getattr(part, "text", None):
            collected.append(part.text)
    return "".join(collected).strip()


ALLOWED_ASPECTS = [
    ("1:1", 1.0),
    ("2:3", 2 / 3),
    ("3:2", 3 / 2),
    ("3:4", 3 / 4),
    ("4:3", 4 / 3),
    ("4:5", 4 / 5),
    ("5:4", 5 / 4),
    ("9:16", 9 / 16),
    ("16:9", 16 / 9),
    ("21:9", 21 / 9),
]


def _auto_aspect_ratio(images: List[torch.Tensor]) -> str:
    for img in images:
        try:
            if img is None:
                continue
            h, w = int(img.shape[1]), int(img.shape[2])  # ComfyUI tensors: [B, H, W, C]
            if h <= 0 or w <= 0:
                continue
            ratio = w / h
            best = min(ALLOWED_ASPECTS, key=lambda x: abs(x[1] - ratio))
            return best[0]
        except Exception:
            continue
    return "1:1"


class Gemini3ProPreviewText:
    """
    Text/multimodal node for gemini-3-pro-preview.
    Accepts an optional pair of images that are sent with media resolution controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "Explain this image", "multiline": True}),
                "model": (["gemini-3-pro-preview"], {"default": "gemini-3-pro-preview"}),
                "media_resolution": (
                    [
                        "auto",
                        "media_resolution_low",
                        "media_resolution_medium",
                        "media_resolution_high",
                    ],
                    {"default": "media_resolution_high"},
                ),
                "thinking_level": (
                    ["default", "low", "high"],
                    {"default": "default"},
                ),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Gemini 3"

    def run(
        self,
        api_key: str,
        prompt: str,
        model: str,
        media_resolution: str,
        thinking_level: str,
        image_1=None,
        image_2=None,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required.")

        key = _require_api_key(api_key)
        client = genai.Client(api_key=key)

        resolution = None if media_resolution == "auto" else media_resolution
        parts: List[types.Part] = [types.Part.from_text(text=prompt)]

        for image in (image_1, image_2):
            if image is not None:
                parts.append(_tensor_to_part(image, resolution))

        config_kwargs = {"response_modalities": ["TEXT"]}
        if thinking_level != "default":
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(**config_kwargs),
        )

        parts_out: List[types.Part] = []
        if getattr(response, "candidates", None):
            first_candidate = response.candidates[0]
            if first_candidate and first_candidate.content:
                parts_out = first_candidate.content.parts or []

        text = (response.text or _gather_text_from_parts(parts_out)).strip()
        return (text,)


class Gemini3ProImagePreview:
    """
    Image generation node for gemini-3-pro-image-preview with aspect ratio and resolution controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "Generate a cinematic landscape", "multiline": True}),
                "model": (["gemini-3-pro-image-preview"], {"default": "gemini-3-pro-image-preview"}),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    {"default": "1:1"},
                ),
                "image_size": (
                    ["1K", "2K", "4K"],
                    {"default": "1K"},
                ),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "run"
    CATEGORY = "Gemini 3"

    def run(
        self,
        api_key: str,
        prompt: str,
        model: str,
        aspect_ratio: str,
        image_size: str,
        reference_image=None,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required.")

        key = _require_api_key(api_key)
        client = genai.Client(api_key=key)

        parts: List[types.Part] = [types.Part.from_text(text=prompt)]
        if reference_image is not None:
            parts.append(_tensor_to_part(reference_image, None))

        chosen_aspect = aspect_ratio
        if aspect_ratio == "auto":
            chosen_aspect = _auto_aspect_ratio([reference_image])

        image_config = types.ImageConfig(aspect_ratio=chosen_aspect, image_size=image_size)
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=image_config,
            ),
        )

        candidates = getattr(response, "candidates", None) or []
        candidate = candidates[0] if candidates else None
        parts_out = candidate.content.parts if candidate and candidate.content else []
        text = (getattr(response, "text", "") or _gather_text_from_parts(parts_out)).strip()

        image_bytes = None
        for part in parts_out:
            if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                image_bytes = part.inline_data.data
                break

        if image_bytes is None:
            raise ValueError("Model did not return an image. Text response: %s" % text)

        image_tensor = _bytes_to_tensor(image_bytes)
        return (image_tensor, text)


NODE_CLASS_MAPPINGS = {
    "Gemini3ProPreviewText": Gemini3ProPreviewText,
    "Gemini3ProImagePreview": Gemini3ProImagePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini3ProPreviewText": "Gemini 3 Pro (Text)",
    "Gemini3ProImagePreview": "Gemini 3 Pro Image",
}
