"""SAM3 backends: mock (default), transformers, or passthrough notes."""

from __future__ import annotations

import os
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
    return os.environ.get("SAM3_BACKEND", "mock").lower()


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


_TRANSFORMERS_MODEL = None


def _get_transformers_model():
    global _TRANSFORMERS_MODEL
    if _TRANSFORMERS_MODEL is not None:
        return _TRANSFORMERS_MODEL
    try:
        # Optional heavy dependency — only used when SAM3_BACKEND=transformers
        from transformers import SamModel, SamProcessor  # type: ignore
        import torch

        model_id = os.environ.get("SAM3_MODEL_ID", "facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained(model_id)
        model = SamModel.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        _TRANSFORMERS_MODEL = (processor, model, device, torch)
        return _TRANSFORMERS_MODEL
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "SAM3_BACKEND=transformers but model load failed. "
            "Install torch+transformers or use SAM3_BACKEND=mock. "
            f"Error: {exc}"
        ) from exc


def _transformers_segment(image_rgb: np.ndarray, prompt: VisualPrompt) -> tuple[np.ndarray, float]:
    processor, model, device, torch = _get_transformers_model()
    seeds = _seed_points(prompt)
    pos = [[x, y] for x, y, p in seeds if p] or [[seeds[0][0], seeds[0][1]]]
    labels = [1] * len(pos)
    for x, y, p in seeds:
        if not p:
            pos.append([x, y])
            labels.append(0)

    inputs = processor(
        images=image_rgb,
        input_points=[pos],
        input_labels=[labels],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    scores = outputs.iou_scores.cpu().numpy()[0]
    best = int(scores.argmax())
    mask = masks[0, best].numpy().astype(bool)
    return mask, float(scores[best])


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
