import base64
import io
import json
import math
import os
import secrets
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from google import genai
from google.genai import types


CONFIG_FILENAME = "config.json"


def _dedupe_keys(values: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for value in values:
        if not isinstance(value, str):
            continue
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


def _load_api_keys_from_config() -> List[str]:
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)
    if not os.path.exists(config_path):
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise ValueError(f"Failed to load {CONFIG_FILENAME}: {exc}") from exc

    keys = data.get("api_keys", [])
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list):
        raise ValueError(f"{CONFIG_FILENAME} api_keys must be a list of strings.")
    return _dedupe_keys(keys)


def _resolve_api_keys(user_key: str) -> List[str]:
    env_key = os.getenv("GEMINI_API_KEY", "")
    keys = _dedupe_keys([user_key] + _load_api_keys_from_config() + [env_key])
    if not keys:
        raise ValueError(
            "API key is required. Provide it in the node, add it to config.json, or set GEMINI_API_KEY."
        )
    return keys


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if any(token in message for token in ("resource_exhausted", "quota", "rate limit", "429")):
        return True

    for attr in ("status_code", "code"):
        code = getattr(exc, attr, None)
        if callable(code):
            try:
                code = code()
            except Exception:
                code = None
        if code == 429 or str(code).upper() == "RESOURCE_EXHAUSTED":
            return True
    return False


def _run_with_key_rotation(api_keys: List[str], request_fn):
    last_quota_error = None
    for key in api_keys:
        client = genai.Client(api_key=key)
        try:
            return request_fn(client)
        except Exception as exc:
            if _is_quota_error(exc):
                last_quota_error = exc
                continue
            raise
    if last_quota_error is not None:
        raise last_quota_error
    raise ValueError("No valid API key available.")


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


def _tensor_to_base64_jpeg(image_tensor: torch.Tensor, quality: int = 75) -> str:
    array = np.clip(image_tensor[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _gather_text_from_parts(parts: List[types.Part]) -> str:
    collected = []
    for part in parts:
        if getattr(part, "text", None):
            collected.append(part.text)
    return "".join(collected).strip()


AZIMUTH_MAP = {
    0: "front view",
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view",
}

ELEVATION_MAP = {
    -30: "low-angle shot",
    0: "eye-level shot",
    30: "elevated shot",
    60: "high-angle shot",
}

DISTANCE_MAP = {
    0.6: "close-up",
    1.0: "medium shot",
    1.8: "wide shot",
}


def _snap_to_nearest(value: float, options: List[float]) -> float:
    return min(options, key=lambda x: abs(x - value))


def _normalize_azimuth(value: float) -> float:
    return float(value) % 360.0


def _build_camera_prompt(azimuth: float, elevation: float, distance: float) -> str:
    azimuth = _normalize_azimuth(azimuth)
    azimuth_snapped = _snap_to_nearest(azimuth, list(AZIMUTH_MAP.keys()))
    elevation_snapped = _snap_to_nearest(float(elevation), list(ELEVATION_MAP.keys()))
    distance_snapped = _snap_to_nearest(float(distance), list(DISTANCE_MAP.keys()))
    return (
        f"{AZIMUTH_MAP[azimuth_snapped]} "
        f"{ELEVATION_MAP[elevation_snapped]} "
        f"{DISTANCE_MAP[distance_snapped]}"
    )


def _supports_field(model_cls, field_name: str) -> bool:
    """
    Best-effort check for whether a genai `types.*` model supports a given field.
    Handles common patterns (pydantic v1/v2, dataclasses). Falls back to False.
    """

    try:
        model_fields = getattr(model_cls, "model_fields", None)
        if isinstance(model_fields, dict):
            return field_name in model_fields

        fields = getattr(model_cls, "__fields__", None)
        if isinstance(fields, dict):
            return field_name in fields

        dataclass_fields = getattr(model_cls, "__dataclass_fields__", None)
        if isinstance(dataclass_fields, dict):
            return field_name in dataclass_fields
    except Exception:
        return False

    return False


INT32_MAX = (2**31) - 1


def _random_seed_int32() -> int:
    # Gemini generation_config.seed is TYPE_INT32 (signed). Keep it in-range.
    return secrets.randbelow(INT32_MAX + 1)


def _normalize_seed_int32(seed: Optional[int], mode: str) -> int:
    parsed_seed = 0 if seed is None else int(seed)
    if mode == "random_if_negative" and parsed_seed < 0:
        return _random_seed_int32()
    if mode == "clamp":
        if parsed_seed < 0:
            return 0
        if parsed_seed > INT32_MAX:
            return INT32_MAX
        return parsed_seed
    return parsed_seed % (INT32_MAX + 1)


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


class Gemini3Camera3DPrompt:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build_prompt"
    CATEGORY = "Gemini3/Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "azimuth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "elevation": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 60.0, "step": 1.0}),
                "distance": ("FLOAT", {"default": 1.0, "min": 0.6, "max": 1.8, "step": 0.05}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    def build_prompt(self, azimuth: float, elevation: float, distance: float, image=None):
        prompt = _build_camera_prompt(azimuth, elevation, distance)
        result = (prompt,)
        if image is None or not isinstance(image, torch.Tensor) or image.numel() == 0:
            return result
        img_base64 = _tensor_to_base64_jpeg(image)
        return {"ui": {"bg_image": [img_base64]}, "result": result}


class Gemini3ProPreviewText:
    """
    Text/multimodal node for gemini-3-pro-preview.
    Accepts an optional pair of images that are sent with media resolution controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional_images = {f"image_{idx}": ("IMAGE",) for idx in range(1, 11)}
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
            "optional": optional_images,
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
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required.")

        api_keys = _resolve_api_keys(api_key)

        resolution = None if media_resolution == "auto" else media_resolution
        parts: List[types.Part] = [types.Part.from_text(text=prompt)]

        for image in (
            image_1,
            image_2,
            image_3,
            image_4,
            image_5,
            image_6,
            image_7,
            image_8,
            image_9,
            image_10,
        ):
            if image is not None:
                parts.append(_tensor_to_part(image, resolution))

        config_kwargs = {"response_modalities": ["TEXT"]}
        if thinking_level != "default":
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

        def _request(client):
            return client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(**config_kwargs),
            )

        response = _run_with_key_rotation(api_keys, _request)

        parts_out: List[types.Part] = []
        if getattr(response, "candidates", None):
            first_candidate = response.candidates[0]
            if first_candidate and first_candidate.content:
                parts_out = first_candidate.content.parts or []

        text = (response.text or _gather_text_from_parts(parts_out)).strip()
        return (text,)


class GeminiSeedInt32:
    """
    Seed helper that normalizes any integer into a Gemini-friendly signed int32.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "mode": (["random_if_negative", "wrap", "clamp"], {"default": "random_if_negative"}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "run"
    CATEGORY = "Gemini 3"

    def run(self, seed: int, mode: str):
        resolved_seed = _normalize_seed_int32(seed, mode)
        return (resolved_seed,)


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
                "seed": ("INT", {"default": -1, "min": -1, "max": INT32_MAX}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "reference_image_5": ("IMAGE",),
                "reference_image_6": ("IMAGE",),
                "reference_image_7": ("IMAGE",),
                "reference_image_8": ("IMAGE",),
                "reference_image_9": ("IMAGE",),
                "reference_image_10": ("IMAGE",),
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
        seed: int,
        reference_image=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        reference_image_5=None,
        reference_image_6=None,
        reference_image_7=None,
        reference_image_8=None,
        reference_image_9=None,
        reference_image_10=None,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required.")

        api_keys = _resolve_api_keys(api_key)

        parts: List[types.Part] = [types.Part.from_text(text=prompt)]
        reference_images = [
            reference_image,
            reference_image_2,
            reference_image_3,
            reference_image_4,
            reference_image_5,
            reference_image_6,
            reference_image_7,
            reference_image_8,
            reference_image_9,
            reference_image_10,
        ]
        for image in reference_images:
            if image is not None:
                parts.append(_tensor_to_part(image, None))

        chosen_aspect = aspect_ratio
        if aspect_ratio == "auto":
            chosen_aspect = _auto_aspect_ratio(reference_images)

        parsed_seed = -1 if seed is None else int(seed)
        if parsed_seed < 0:
            resolved_seed = _random_seed_int32()
        elif parsed_seed > INT32_MAX:
            raise ValueError(f"Seed must be <= {INT32_MAX} (Gemini expects signed int32).")
        else:
            resolved_seed = parsed_seed

        image_config_kwargs = {"aspect_ratio": chosen_aspect, "image_size": image_size}
        if _supports_field(types.ImageConfig, "seed"):
            image_config_kwargs["seed"] = resolved_seed
        image_config = types.ImageConfig(**image_config_kwargs)

        config_kwargs = {"response_modalities": ["IMAGE", "TEXT"], "image_config": image_config}
        if _supports_field(types.GenerateContentConfig, "seed"):
            config_kwargs["seed"] = resolved_seed

        def _request(client):
            return client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(**config_kwargs),
            )

        response = _run_with_key_rotation(api_keys, _request)

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
    "Gemini3Camera3DPrompt": Gemini3Camera3DPrompt,
    "Gemini3ProPreviewText": Gemini3ProPreviewText,
    "GeminiSeedInt32": GeminiSeedInt32,
    "Gemini3ProImagePreview": Gemini3ProImagePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini3Camera3DPrompt": "3D Camera Prompt",
    "Gemini3ProPreviewText": "Gemini 3 Pro (Text)",
    "GeminiSeedInt32": "Gemini Seed (int32)",
    "Gemini3ProImagePreview": "Gemini 3 Pro Image",
}
