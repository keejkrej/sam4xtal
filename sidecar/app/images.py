"""Image decode helpers and embedding cache."""

from __future__ import annotations

import base64
import hashlib
import io
import uuid
from dataclasses import dataclass
from typing import Optional

import cv2
import httpx
import numpy as np
from PIL import Image

from .schemas import InferenceRequestImage


@dataclass
class CachedEmbedding:
    image_id: str
    image_rgb: np.ndarray
    # Reserved for real SAM3 embeddings
    embedding: Optional[np.ndarray] = None


_CACHE: dict[str, CachedEmbedding] = {}


def decode_image(image: InferenceRequestImage) -> np.ndarray:
    if image.type == "url":
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(image.value)
            resp.raise_for_status()
            data = resp.content
    else:
        raw = image.value
        if "," in raw and raw.strip().startswith("data:"):
            raw = raw.split(",", 1)[1]
        data = base64.b64decode(raw)

    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        # Fallback via Pillow
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(pil)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_content_id(image: InferenceRequestImage) -> str:
    payload = f"{image.type}:{image.value[:256]}:{len(image.value)}".encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def store_image(
    image_rgb: np.ndarray,
    image_id: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,
) -> CachedEmbedding:
    eid = image_id or str(uuid.uuid4())
    cached = CachedEmbedding(image_id=eid, image_rgb=image_rgb, embedding=embedding)
    _CACHE[eid] = cached
    return cached


def get_cached(image_id: str) -> Optional[CachedEmbedding]:
    return _CACHE.get(image_id)


def clear_cache() -> None:
    _CACHE.clear()
