"""SAM3 backends: mock, transformers (Tracker PVS + Sam3Model PCS)."""

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
    Sam3Prompt,
    SegmentationPrediction,
    SegmentationResponse,
    VisualPrompt,
    VisualSegmentRequest,
)


def _backend_name() -> str:
    return os.environ.get("SAM3_BACKEND", "transformers").lower()


# PVS (point/box interactive) — Sam3TrackerModel
_TRANSFORMERS_MODEL = None
# PCS (text / image-exemplar concept transfer) — Sam3Model
_CONCEPT_MODEL = None
_LOAD_LOCK = threading.Lock()
_LOAD_STATE = "idle"  # idle | loading | ready | error
_LOAD_ERROR: str | None = None
_CONCEPT_LOAD_STATE = "idle"  # idle | loading | ready | error
_CONCEPT_LOAD_ERROR: str | None = None


class ModelNotReady(RuntimeError):
    """Raised while SAM3 weights are still downloading / loading."""

    def __init__(self, message: str, *, state: str = "loading") -> None:
        super().__init__(message)
        self.state = state


def backend_status() -> dict[str, Any]:
    name = _backend_name()
    tracker_ready = name != "transformers" or _TRANSFORMERS_MODEL is not None
    concept_ready = name != "transformers" or _CONCEPT_MODEL is not None
    status: dict[str, Any] = {
        "backend": name,
        "ready": tracker_ready and _LOAD_STATE != "error",
        "load_state": ("ready" if name == "mock" else _LOAD_STATE),
        "model_loaded": _TRANSFORMERS_MODEL is not None or name == "mock",
        "concept_ready": concept_ready and _CONCEPT_LOAD_STATE != "error",
        "concept_load_state": (
            "ready" if name == "mock" else _CONCEPT_LOAD_STATE
        ),
        "concept_model_loaded": _CONCEPT_MODEL is not None or name == "mock",
        "device": None,
        "model_id": os.environ.get("SAM3_MODEL_ID", "facebook/sam3"),
        "error": _LOAD_ERROR,
        "concept_error": _CONCEPT_LOAD_ERROR,
    }
    if _TRANSFORMERS_MODEL is not None:
        _processor, _model, device, _torch = _TRANSFORMERS_MODEL
        status["device"] = device
    elif _CONCEPT_MODEL is not None:
        _processor, _model, device, _torch = _CONCEPT_MODEL
        status["device"] = device
    return status


def ensure_ready() -> None:
    """Ensure Tracker (PVS) model is usable."""
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
    start_backend_load()
    raise ModelNotReady(
        "SAM3 model is still downloading or loading onto the GPU. "
        "Retry in a few seconds.",
        state="loading",
    )


def ensure_concept_ready() -> None:
    """Ensure Sam3Model (PCS / concept transfer) is loaded."""
    if _backend_name() != "transformers":
        return
    if _CONCEPT_MODEL is not None:
        return
    if _CONCEPT_LOAD_STATE == "error":
        raise ModelNotReady(
            _CONCEPT_LOAD_ERROR or "SAM3 concept model failed to load",
            state="error",
        )
    if _CONCEPT_LOAD_STATE == "loading":
        raise ModelNotReady(
            "SAM3 concept model is still downloading or loading onto the GPU. "
            "Retry in a few seconds.",
            state="loading",
        )
    start_concept_model_load()
    raise ModelNotReady(
        "SAM3 concept model is still downloading or loading onto the GPU. "
        "Retry in a few seconds.",
        state="loading",
    )


def _load_transformers_worker() -> None:
    global _LOAD_STATE, _LOAD_ERROR
    try:
        print("[sam4xtal] loading facebook/sam3 Tracker (PVS) …", flush=True)
        _get_transformers_model()
        status = backend_status()
        _LOAD_STATE = "ready"
        _LOAD_ERROR = None
        print(
            f"[sam4xtal] tracker backend={status['backend']} "
            f"model_loaded={status['model_loaded']} "
            f"device={status['device']} model_id={status['model_id']}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        _LOAD_STATE = "error"
        _LOAD_ERROR = str(exc)
        print(f"[sam4xtal] tracker model load failed: {exc}", flush=True)


def _load_concept_worker() -> None:
    global _CONCEPT_LOAD_STATE, _CONCEPT_LOAD_ERROR
    try:
        print("[sam4xtal] loading facebook/sam3 Model (PCS) …", flush=True)
        _get_concept_model()
        _CONCEPT_LOAD_STATE = "ready"
        _CONCEPT_LOAD_ERROR = None
        status = backend_status()
        print(
            f"[sam4xtal] concept backend={status['backend']} "
            f"concept_model_loaded={status['concept_model_loaded']} "
            f"device={status['device']} model_id={status['model_id']}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        _CONCEPT_LOAD_STATE = "error"
        _CONCEPT_LOAD_ERROR = str(exc)
        print(f"[sam4xtal] concept model load failed: {exc}", flush=True)


def start_backend_load() -> dict[str, Any]:
    """Start Tracker model download/load in a background thread."""
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
            name="sam3-tracker-load",
            daemon=True,
        )
        thread.start()
    return backend_status()


def start_concept_model_load() -> dict[str, Any]:
    """Start PCS (Sam3Model) load in a background thread."""
    global _CONCEPT_LOAD_STATE, _CONCEPT_LOAD_ERROR
    if _backend_name() != "transformers":
        _CONCEPT_LOAD_STATE = "ready"
        _CONCEPT_LOAD_ERROR = None
        return backend_status()

    with _LOAD_LOCK:
        if _CONCEPT_MODEL is not None:
            _CONCEPT_LOAD_STATE = "ready"
            return backend_status()
        if _CONCEPT_LOAD_STATE == "loading":
            return backend_status()
        _CONCEPT_LOAD_STATE = "loading"
        _CONCEPT_LOAD_ERROR = None
        thread = threading.Thread(
            target=_load_concept_worker,
            name="sam3-concept-load",
            daemon=True,
        )
        thread.start()
    return backend_status()


def warm_backend() -> dict[str, Any]:
    """Compatibility alias — starts Tracker background load without blocking."""
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
    else:
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
        seeds.append((int(round(prompt.box.x)), int(round(prompt.box.y)), True))
    return seeds


def _cleanup_enabled() -> bool:
    raw = os.environ.get("SAM3_MASK_CLEANUP", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _open_kernel_size() -> int:
    raw = os.environ.get("SAM3_MASK_OPEN_K", "5").strip()
    try:
        k = int(raw)
    except ValueError:
        k = 5
    if k <= 0:
        return 0
    return k if k % 2 == 1 else k + 1


def _largest_component(labels: np.ndarray, n_labels: int) -> np.ndarray:
    if n_labels <= 1:
        return np.zeros(labels.shape, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def _cleanup_mask(mask: np.ndarray) -> np.ndarray:
    """Drop leak islands / thin bridges; keep a single connected region."""
    if not _cleanup_enabled() or not mask.any():
        return mask.astype(bool)

    opened = mask.astype(np.uint8)
    k = _open_kernel_size()
    if k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel)

    if not opened.any():
        n_raw, labels_raw = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return _largest_component(labels_raw, n_raw)

    n_labels, labels = cv2.connectedComponents(opened, connectivity=8)
    return _largest_component(labels, n_labels)


def _mock_segment(image_rgb: np.ndarray, prompt: VisualPrompt) -> tuple[np.ndarray, float]:
    """Region-growing mock mask from click points — good enough for UI wiring."""
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    seeds = _seed_points(prompt)
    if not seeds:
        return np.zeros((h, w), dtype=bool), 0.0

    pos = [(x, y) for x, y, posi in seeds if posi]
    neg = [(x, y) for x, y, posi in seeds if not posi]
    if not pos:
        pos = [(seeds[0][0], seeds[0][1])]

    sx, sy = pos[0]
    sx = int(np.clip(sx, 0, w - 1))
    sy = int(np.clip(sy, 0, h - 1))

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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    if neg:
        region_u8 = region.astype(np.uint8)
        for nx, ny in neg:
            if 0 <= nx < w and 0 <= ny < h:
                cv2.circle(region_u8, (nx, ny), 12, 0, -1)
        region = region_u8.astype(bool)

    if region.mean() > 0.45:
        region = np.zeros((h, w), dtype=bool)
        radius = max(20, min(h, w) // 8)
        yy, xx = np.ogrid[:h, :w]
        for x, y in pos:
            region |= (xx - x) ** 2 + (yy - y) ** 2 <= radius**2

    confidence = 0.75 if region.any() else 0.0
    return region, confidence


def _box_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _clip_xyxy(box: list[float], w: int, h: int) -> list[float]:
    x0 = float(np.clip(box[0], 0, w - 1))
    y0 = float(np.clip(box[1], 0, h - 1))
    x1 = float(np.clip(box[2], x0 + 1, w))
    y1 = float(np.clip(box[3], y0 + 1, h))
    return [x0, y0, x1, y1]


def _mock_concept_pcs(
    image_rgb: np.ndarray,
    boxes_xyxy: list[list[float]],
    labels: list[int],
    thresh: float,
) -> list[tuple[np.ndarray, float]]:
    """Mock PCS: template-match positive exemplar crops → similar regions."""
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    pos_boxes = [
        _clip_xyxy(b, w, h)
        for b, lab in zip(boxes_xyxy, labels)
        if lab != 0
    ]
    neg_boxes = [
        _clip_xyxy(b, w, h)
        for b, lab in zip(boxes_xyxy, labels)
        if lab == 0
    ]
    if not pos_boxes:
        return []

    # Score map: max correlation over positive templates (multi-scale light)
    score = np.zeros((h, w), dtype=np.float32)
    for box in pos_boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        tmpl = gray[y0:y1, x0:x1]
        if tmpl.size < 9 or tmpl.shape[0] < 3 or tmpl.shape[1] < 3:
            continue
        th, tw = tmpl.shape[:2]
        for scale in (0.75, 1.0, 1.25):
            nh, nw = max(3, int(th * scale)), max(3, int(tw * scale))
            if nh >= h or nw >= w:
                continue
            scaled = cv2.resize(tmpl, (nw, nh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
            # paste into full-size score at template center
            pad_y, pad_x = nh // 2, nw // 2
            rh, rw = res.shape
            y_end, x_end = pad_y + rh, pad_x + rw
            if y_end > h or x_end > w:
                rh = min(rh, h - pad_y)
                rw = min(rw, w - pad_x)
                res = res[:rh, :rw]
                y_end, x_end = pad_y + rh, pad_x + rw
            patch = score[pad_y:y_end, pad_x:x_end]
            if patch.shape == res.shape:
                np.maximum(patch, res, out=patch)

    # Suppress negatives
    for box in neg_boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        score[y0:y1, x0:x1] = 0.0

    # Peak picking → instance masks (approx ellipses around peaks)
    work = score.copy()
    results: list[tuple[np.ndarray, float]] = []
    min_dist = max(8, int(min(h, w) * 0.03))
    peak_floor = max(float(thresh), 0.35)

    for _ in range(64):
        min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(work)
        conf = float(max_val)
        if conf < peak_floor:
            break
        cx, cy = int(max_loc[0]), int(max_loc[1])
        # Estimate size from first positive box
        ref = pos_boxes[0]
        rw = max(8, int((ref[2] - ref[0]) / 2))
        rh = max(8, int((ref[3] - ref[1]) / 2))
        mask = np.zeros((h, w), dtype=bool)
        yy, xx = np.ogrid[:h, :w]
        mask |= ((xx - cx) / rw) ** 2 + ((yy - cy) / rh) ** 2 <= 1.0
        # zero neighborhood so we don't re-pick
        y0, y1 = max(0, cy - min_dist), min(h, cy + min_dist + 1)
        x0, x1 = max(0, cx - min_dist), min(w, cx + min_dist + 1)
        work[y0:y1, x0:x1] = 0.0
        if conf >= thresh and mask.any():
            results.append((mask, conf))

    # Always include flood-fill from positive exemplar centers as anchors
    for box in pos_boxes:
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        vp = VisualPrompt(points=[Point(x=cx, y=cy, positive=True)])
        m, c = _mock_segment(image_rgb, vp)
        if m.any():
            # avoid near-duplicates
            if not any(_box_iou(
                [float(np.where(m)[1].min()), float(np.where(m)[0].min()),
                 float(np.where(m)[1].max() + 1), float(np.where(m)[0].max() + 1)],
                [float(np.where(rm)[1].min()), float(np.where(rm)[0].min()),
                 float(np.where(rm)[1].max() + 1), float(np.where(rm)[0].max() + 1)],
            ) > 0.5 for rm, _ in results if rm.any()):
                results.append((m, max(c, 0.8)))

    return results


def _get_transformers_model():
    global _TRANSFORMERS_MODEL
    if _TRANSFORMERS_MODEL is not None:
        return _TRANSFORMERS_MODEL
    try:
        # SAM 3 Tracker = point/box PVS (interactive clicks).
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
            "SAM3_BACKEND=transformers but facebook/sam3 Tracker load failed. "
            "Accept the model license on Hugging Face and set HF_TOKEN if gated, "
            "or use SAM3_BACKEND=mock / .\\run.ps1 --mock. "
            f"Error: {exc}"
        ) from exc


def _get_concept_model():
    """Sam3Model + Sam3Processor for PCS (text / image-exemplar concept transfer)."""
    global _CONCEPT_MODEL
    if _CONCEPT_MODEL is not None:
        return _CONCEPT_MODEL
    try:
        from transformers import Sam3Model, Sam3Processor  # type: ignore
        import torch

        model_id = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
        processor = Sam3Processor.from_pretrained(model_id)
        model = Sam3Model.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        _CONCEPT_MODEL = (processor, model, device, torch)
        return _CONCEPT_MODEL
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "SAM3 concept model (Sam3Model) load failed. "
            "Concept transfer needs the PCS head, not just the Tracker. "
            "Accept the model license on Hugging Face and set HF_TOKEN if gated, "
            "or use SAM3_BACKEND=mock. "
            f"Error: {exc}"
        ) from exc


def _transformers_segment(image_rgb: np.ndarray, prompt: VisualPrompt) -> tuple[np.ndarray, float]:
    from PIL import Image

    processor, model, device, torch = _get_transformers_model()
    seeds = _seed_points(prompt)
    if not seeds:
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), dtype=bool), 0.0

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


def _transformers_concept_pcs(
    image_rgb: np.ndarray,
    boxes_xyxy: list[list[float]],
    labels: list[int],
    text: str | None,
    output_prob_thresh: float,
    mask_threshold: float = 0.5,
) -> list[tuple[np.ndarray, float]]:
    """Roboflow-compatible PCS via Sam3Model (text and/or image exemplars)."""
    from PIL import Image

    processor, model, device, torch = _get_concept_model()
    h, w = image_rgb.shape[:2]
    clipped = [_clip_xyxy(b, w, h) for b in boxes_xyxy]

    pil = Image.fromarray(image_rgb)
    kwargs: dict[str, Any] = {
        "images": pil,
        "return_tensors": "pt",
    }
    if clipped:
        # HF: pixel xyxy, batch layout [image][boxes]
        kwargs["input_boxes"] = [clipped]
        kwargs["input_boxes_labels"] = [labels if labels else [1] * len(clipped)]
    if text and text.strip():
        kwargs["text"] = text.strip()
    if not clipped and not (text and text.strip()):
        raise ValueError("PCS requires text and/or exemplar boxes")

    inputs = processor(**kwargs)
    inputs = {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = inputs.get("original_sizes")
    if target_sizes is not None:
        target_sizes = target_sizes.tolist()
    else:
        target_sizes = [[h, w]]

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=output_prob_thresh,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )[0]

    out: list[tuple[np.ndarray, float]] = []
    masks = results.get("masks")
    scores = results.get("scores")
    if masks is None:
        return out

    n = len(masks) if not hasattr(masks, "shape") else int(masks.shape[0])
    for i in range(n):
        m = masks[i]
        if hasattr(m, "cpu"):
            m = m.cpu().numpy()
        mask = np.asarray(m).astype(bool)
        if mask.ndim > 2:
            mask = mask.squeeze()
        conf = 1.0
        if scores is not None:
            s = scores[i]
            conf = float(s.item() if hasattr(s, "item") else s)
        if conf < output_prob_thresh or not mask.any():
            continue
        mask = _cleanup_mask(mask)
        if mask.any():
            out.append((mask, conf))
    return out


def _run_pcs_prompt(
    image_rgb: np.ndarray,
    prompt: Sam3Prompt,
    default_thresh: float,
    fmt: str,
) -> list[SegmentationPrediction]:
    """Run one Roboflow Sam3Prompt (text / visual exemplars / both)."""
    boxes = prompt.boxes_xyxy()
    labels = prompt.labels_int() if boxes else []
    text = prompt.text.strip() if prompt.text and prompt.text.strip() else None
    thresh = (
        prompt.output_prob_thresh
        if prompt.output_prob_thresh is not None
        else default_thresh
    )
    backend = _backend_name()

    if backend == "transformers":
        ensure_concept_ready()
        pairs = _transformers_concept_pcs(
            image_rgb, boxes, labels, text, thresh, 0.5
        )
    elif boxes:
        pairs = _mock_concept_pcs(image_rgb, boxes, labels, thresh)
    else:
        # text-only mock: soft center blob
        h, w = image_rgb.shape[:2]
        vp = VisualPrompt(points=[Point(x=w // 2, y=h // 2, positive=True)])
        mask, conf = _mock_segment(image_rgb, vp)
        pairs = [(_cleanup_mask(mask), conf)] if conf >= thresh else []

    return [_prediction_from_mask(m, c, fmt) for m, c in pairs]


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

    backend = _backend_name()
    results: list[PromptResult] = []
    all_preds: list[SegmentationPrediction] = []

    for idx, prompt in enumerate(prompts):
        if backend == "transformers":
            mask, conf = _transformers_segment(cached.image_rgb, prompt)
        else:
            mask, conf = _mock_segment(cached.image_rgb, prompt)
        mask = _cleanup_mask(mask)
        pred = _prediction_from_mask(mask, conf, req.format)
        results.append(PromptResult(prompt_index=idx, predictions=[pred]))
        all_preds.append(pred)

    elapsed = time.perf_counter() - t0
    return SegmentationResponse(
        time=elapsed,
        prompt_results=results,
        predictions=all_preds,
    )


def concept_segment(req: ConceptSegmentRequest) -> SegmentationResponse:
    """Roboflow-compatible PCS: text and/or image-exemplar prompts.

    Few-shot “a few corrected boxes → all similar instances” is a visual
    prompt with ``boxes`` + ``box_labels`` (same image only).
    """
    t0 = time.perf_counter()
    cached = resolve_image(req.image, req.image_id)
    results: list[PromptResult] = []
    all_preds: list[SegmentationPrediction] = []

    for idx, prompt in enumerate(req.prompts):
        preds = _run_pcs_prompt(
            cached.image_rgb,
            prompt,
            req.output_prob_thresh,
            req.format,
        )
        results.append(
            PromptResult(
                prompt_index=idx,
                predictions=preds,
                echo={
                    "prompt_index": idx,
                    "type": prompt.type,
                    "text": prompt.text,
                    "num_boxes": len(prompt.boxes or []),
                    "box_labels": prompt.labels_int() if prompt.boxes else None,
                },
            )
        )
        all_preds.extend(preds)

    return SegmentationResponse(
        time=time.perf_counter() - t0,
        prompt_results=results,
        predictions=all_preds,
    )
