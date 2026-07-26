"""SAM3 backends: mock, transformers (SAM 3 Tracker), or passthrough notes."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import cv2
import numpy as np

from .images import CachedEmbedding, decode_image, get_cached, store_image
from .schemas import (
    ConceptSegmentRequest,
    EmbedImageRequest,
    EmbedImageResponse,
    InferenceRequestImage,
    Point,
    PromptResult,
    SegmentationPrediction,
    SegmentationResponse,
    VisualPrompt,
    VisualSegmentRequest,
)


def _backend_name() -> str:
    return os.environ.get("SAM3_BACKEND", "transformers").lower()


_TRANSFORMERS_MODEL = None
_LOAD_LOCK = threading.Lock()
_LOAD_STATE = "idle"  # idle | loading | ready | error
_LOAD_ERROR: str | None = None


class ModelNotReady(RuntimeError):
    """Raised while SAM3 weights are still downloading / loading."""

    def __init__(self, message: str, *, state: str = "loading") -> None:
        super().__init__(message)
        self.state = state


def backend_status() -> dict[str, Any]:
    name = _backend_name()
    ready = name != "transformers" or _TRANSFORMERS_MODEL is not None
    status: dict[str, Any] = {
        "backend": name,
        "ready": ready and _LOAD_STATE != "error",
        "load_state": ("ready" if name == "mock" else _LOAD_STATE),
        "model_loaded": _TRANSFORMERS_MODEL is not None or name == "mock",
        "device": None,
        "model_id": os.environ.get("SAM3_MODEL_ID", "facebook/sam3"),
        "error": _LOAD_ERROR,
    }
    if _TRANSFORMERS_MODEL is not None:
        _processor, _model, device, _torch = _TRANSFORMERS_MODEL
        status["device"] = device
    return status


def ensure_ready() -> None:
    """Block callers until the model is usable, or raise ModelNotReady."""
    if _backend_name() != "transformers":
        return
    if _TRANSFORMERS_MODEL is not None:
        return
    if _LOAD_STATE == "error":
        raise ModelNotReady(
            _LOAD_ERROR or "SAM3 model failed to load",
            state="error",
        )
    if _LOAD_STATE == "loading":
        raise ModelNotReady(
            "SAM3 model is still downloading or loading onto the GPU. "
            "Retry in a few seconds.",
            state="loading",
        )
    # idle but not loaded — kick off load and report loading
    start_backend_load()
    raise ModelNotReady(
        "SAM3 model is still downloading or loading onto the GPU. "
        "Retry in a few seconds.",
        state="loading",
    )


def _load_transformers_worker() -> None:
    global _LOAD_STATE, _LOAD_ERROR
    try:
        print("[sam4xtal] loading facebook/sam3 …", flush=True)
        _get_transformers_model()
        status = backend_status()
        _LOAD_STATE = "ready"
        _LOAD_ERROR = None
        print(
            f"[sam4xtal] backend={status['backend']} "
            f"model_loaded={status['model_loaded']} "
            f"device={status['device']} model_id={status['model_id']}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        _LOAD_STATE = "error"
        _LOAD_ERROR = str(exc)
        print(f"[sam4xtal] model load failed: {exc}", flush=True)


def start_backend_load() -> dict[str, Any]:
    """Start model download/load in a background thread (non-blocking)."""
    global _LOAD_STATE, _LOAD_ERROR
    if _backend_name() != "transformers":
        _LOAD_STATE = "ready"
        _LOAD_ERROR = None
        return backend_status()

    with _LOAD_LOCK:
        if _TRANSFORMERS_MODEL is not None:
            _LOAD_STATE = "ready"
            return backend_status()
        if _LOAD_STATE == "loading":
            return backend_status()
        _LOAD_STATE = "loading"
        _LOAD_ERROR = None
        thread = threading.Thread(
            target=_load_transformers_worker,
            name="sam3-model-load",
            daemon=True,
        )
        thread.start()
    return backend_status()


def warm_backend() -> dict[str, Any]:
    """Compatibility alias — starts background load without blocking."""
    return start_backend_load()


def _mask_to_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        pts = contour.reshape(-1, 2).astype(float)
        polygons.append([[float(x), float(y)] for x, y in pts])
    return polygons


def _mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """COCO-style Fortran-order RLE."""
    flat = np.asfortranarray(mask.astype(np.uint8)).ravel(order="F")
    diff = np.diff(flat)
    changes = np.where(diff != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [flat.size]])
    counts: list[int] = []
    # Ensure RLE starts with zeros count
    if flat.size and flat[0] == 1:
        counts.append(0)
    for s, e in zip(starts, ends):
        counts.append(int(e - s))
    h, w = mask.shape
    return {"size": [int(h), int(w)], "counts": counts}


def _prediction_from_mask(
    mask: np.ndarray,
    confidence: float,
    fmt: str,
) -> SegmentationPrediction:
    area = int(mask.sum())
    ys, xs = np.where(mask)
    if len(xs) == 0:
        bbox = [0.0, 0.0, 0.0, 0.0]
    else:
        bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

    if fmt in ("json", "polygon"):
        masks: Any = _mask_to_polygons(mask)
        out_fmt = "polygon" if fmt == "polygon" else "json"
    elif fmt == "rle":
        masks = [_mask_to_rle(mask)]
        out_fmt = "rle"
    else:  # binary — return flattened list (JSON-safe)
        masks = [mask.astype(np.uint8).tolist()]
        out_fmt = "binary"

    return SegmentationPrediction(
        format=out_fmt,
        confidence=float(confidence),
        masks=masks,
        area_px=area,
        bbox_xyxy=bbox,
    )


def _seed_points(prompt: VisualPrompt) -> list[tuple[int, int, bool]]:
    seeds: list[tuple[int, int, bool]] = []
    if prompt.points:
        for p in prompt.points:
            seeds.append((int(round(p.x)), int(round(p.y)), bool(p.positive)))
    if prompt.box is not None:
        # Center-anchored XYWH → use center as a positive seed
        seeds.append((int(round(prompt.box.x)), int(round(prompt.box.y)), True))
    return seeds


def _mock_segment(image_rgb: np.ndarray, prompt: VisualPrompt) -> tuple[np.ndarray, float]:
    """Region-growing mock mask from click points — good enough for UI wiring."""
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    seeds = _seed_points(prompt)
    if not seeds:
        return np.zeros((h, w), dtype=bool), 0.0

    # Use first positive seed for flood fill tolerance
    pos = [(x, y) for x, y, posi in seeds if posi]
    neg = [(x, y) for x, y, posi in seeds if not posi]
    if not pos:
        pos = [(seeds[0][0], seeds[0][1])]

    sx, sy = pos[0]
    sx = int(np.clip(sx, 0, w - 1))
    sy = int(np.clip(sy, 0, h - 1))

    # Adaptive flood fill from seed intensity
    seed_val = int(gray[sy, sx])
    lo = max(5, seed_val // 6)
    hi = max(10, (255 - seed_val) // 5)

    flood = gray.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(
        flood,
        mask,
        (sx, sy),
        255,
        loDiff=lo,
        upDiff=hi,
        flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY,
    )
    region = mask[1:-1, 1:-1] > 0

    # Expand slightly with morphology for porous SEM crystals
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    if neg:
        region_u8 = region.astype(np.uint8)
        for nx, ny in neg:
            if 0 <= nx < w and 0 <= ny < h:
                cv2.circle(region_u8, (nx, ny), 12, 0, -1)
        region = region_u8.astype(bool)

    # If flood filled almost everything, fall back to circular blob
    if region.mean() > 0.45:
        region = np.zeros((h, w), dtype=bool)
        radius = max(20, min(h, w) // 8)
        yy, xx = np.ogrid[:h, :w]
        for x, y in pos:
            region |= (xx - x) ** 2 + (yy - y) ** 2 <= radius**2

    confidence = 0.75 if region.any() else 0.0
    return region, confidence


def _get_transformers_model():
    global _TRANSFORMERS_MODEL
    if _TRANSFORMERS_MODEL is not None:
        return _TRANSFORMERS_MODEL
    try:
        # SAM 3 Tracker = point/box PVS (drop-in for interactive clicks).
        # Sam3Model is text/concept PCS — not what the UI point prompts need.
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor  # type: ignore
        import torch

        model_id = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
        processor = Sam3TrackerProcessor.from_pretrained(model_id)
        model = Sam3TrackerModel.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        _TRANSFORMERS_MODEL = (processor, model, device, torch)
        return _TRANSFORMERS_MODEL
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "SAM3_BACKEND=transformers but facebook/sam3 load failed. "
            "Accept the model license on Hugging Face and set HF_TOKEN if gated, "
            "or use SAM3_BACKEND=mock / .\\run.ps1 --mock. "
            f"Error: {exc}"
        ) from exc


def _transformers_segment(image_rgb: np.ndarray, prompt: VisualPrompt) -> tuple[np.ndarray, float]:
    from PIL import Image

    processor, model, device, torch = _get_transformers_model()
    seeds = _seed_points(prompt)
    if not seeds:
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), dtype=bool), 0.0

    # Sam3Tracker point layout: [image][object][point][xy]
    points: list[list[float]] = []
    labels: list[int] = []
    for x, y, positive in seeds:
        points.append([float(x), float(y)])
        labels.append(1 if positive else 0)
    if not any(labels):
        labels[0] = 1

    pil = Image.fromarray(image_rgb)
    inputs = processor(
        images=pil,
        input_points=[[points]],
        input_labels=[[labels]],
        return_tensors="pt",
    )
    inputs = {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
    )[0]
    # masks: [num_objects, num_masks, H, W]
    scores = getattr(outputs, "iou_scores", None)
    if scores is not None:
        score_np = scores.cpu().numpy()[0, 0]
        best = int(np.argmax(score_np))
        conf = float(score_np[best])
    else:
        best = 0
        conf = 1.0
    mask = masks[0, best].numpy().astype(bool)
    return mask, conf


def resolve_image(
    image: InferenceRequestImage | None,
    image_id: str | None,
) -> CachedEmbedding:
    if image_id:
        cached = get_cached(image_id)
        if cached is not None:
            return cached
        if image is None:
            raise KeyError(f"Unknown image_id: {image_id}")
    if image is None:
        raise ValueError("Either image or a valid image_id is required")
    rgb = decode_image(image)
    return store_image(rgb, image_id=image_id)


def embed_image(req: EmbedImageRequest) -> EmbedImageResponse:
    t0 = time.perf_counter()
    rgb = decode_image(req.image)
    cached = store_image(rgb, image_id=req.image_id)
    h, w = rgb.shape[:2]
    return EmbedImageResponse(
        image_id=cached.image_id,
        image_shape=[h, w, 3],
        time=time.perf_counter() - t0,
    )


def visual_segment(req: VisualSegmentRequest) -> SegmentationResponse:
    ensure_ready()
    t0 = time.perf_counter()
    cached = resolve_image(req.image, req.image_id)
    prompts = req.normalized_prompts()
    if not prompts:
        raise ValueError("At least one visual prompt is required")

    # Roboflow PVS: one prediction returned (best mask for first prompt)
    prompt = prompts[0]
    backend = _backend_name()
    if backend == "transformers":
        mask, conf = _transformers_segment(cached.image_rgb, prompt)
    else:
        mask, conf = _mock_segment(cached.image_rgb, prompt)

    pred = _prediction_from_mask(mask, conf, req.format)
    result = PromptResult(prompt_index=0, predictions=[pred])
    elapsed = time.perf_counter() - t0
    return SegmentationResponse(
        time=elapsed,
        prompt_results=[result],
        predictions=[pred],
    )


def concept_segment(req: ConceptSegmentRequest) -> SegmentationResponse:
    """Stub PCS: treat image center as a soft blob per prompt (mock)."""
    t0 = time.perf_counter()
    cached = resolve_image(req.image, req.image_id)
    h, w = cached.image_rgb.shape[:2]
    results: list[PromptResult] = []
    all_preds: list[SegmentationPrediction] = []

    for idx, prompt in enumerate(req.prompts):
        # Use a center point + optional first box center
        cx, cy = w // 2, h // 2
        if prompt.boxes:
            b = prompt.boxes[0]
            if isinstance(b, dict):
                if "x0" in b:
                    cx = int((b["x0"] + b["x1"]) / 2)
                    cy = int((b["y0"] + b["y1"]) / 2)
                else:
                    cx = int(b.get("x", cx) + b.get("width", 0) / 2)
                    cy = int(b.get("y", cy) + b.get("height", 0) / 2)
        vp = VisualPrompt(points=[Point(x=cx, y=cy, positive=True)])
        mask, conf = _mock_segment(cached.image_rgb, vp)
        if conf < req.output_prob_thresh:
            preds: list[SegmentationPrediction] = []
        else:
            preds = [_prediction_from_mask(mask, conf, req.format)]
        results.append(
            PromptResult(
                prompt_index=idx,
                predictions=preds,
                echo={
                    "prompt_index": idx,
                    "type": prompt.type,
                    "text": prompt.text,
                    "num_boxes": len(prompt.boxes or []),
                },
            )
        )
        all_preds.extend(preds)

    return SegmentationResponse(
        time=time.perf_counter() - t0,
        prompt_results=results,
        predictions=all_preds,
    )
