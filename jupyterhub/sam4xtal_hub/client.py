"""Minimal HTTP client for the sam4xtal SAM3 sidecar (JupyterHub path)."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


class SidecarError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class SidecarClient:
    def __init__(self, base_url: str = "http://127.0.0.1:9001", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = raw
            detail = payload
            if isinstance(payload, dict):
                detail = payload.get("detail", payload)
            raise SidecarError(
                f"{method} {path} → HTTP {e.code}: {detail}",
                status=e.code,
                body=payload,
            ) from e
        except urllib.error.URLError as e:
            raise SidecarError(
                f"sidecar unreachable at {self.base_url}: {e.reason}"
            ) from e

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def wait_ready(self, timeout_s: float = 600.0, poll_s: float = 2.0) -> dict[str, Any]:
        """Poll /health until tracker is ready (or mock)."""
        deadline = time.time() + timeout_s
        last: dict[str, Any] = {}
        while time.time() < deadline:
            try:
                last = self.health()
            except SidecarError as e:
                # Process up but model still loading often returns 503.
                if e.status in (None, 503):
                    last = e.body if isinstance(e.body, dict) else {"detail": str(e)}
                    time.sleep(poll_s)
                    continue
                raise

            state = last.get("load_state")
            if state == "error":
                raise SidecarError(
                    f"model load failed: {last.get('error')}",
                    body=last,
                )
            if state == "loading":
                time.sleep(poll_s)
                continue
            # ready | mock | idle-after-load
            if (
                last.get("backend") == "mock"
                or state == "ready"
                or last.get("model_loaded")
                or last.get("ready")
            ):
                return last
            time.sleep(poll_s)
        raise SidecarError(f"timed out waiting for sidecar ({timeout_s}s): {last}")

    def visual_segment(
        self,
        image_b64: str,
        points: list[dict[str, Any]],
        *,
        multimask_output: bool = False,
        fmt: str = "binary",
    ) -> dict[str, Any]:
        """Point-prompted segmentation. Default binary masks for easy numpy use."""
        body = {
            "image": {"type": "base64", "value": image_b64},
            "prompts": [{"points": points}],
            "multimask_output": multimask_output,
            "format": fmt,
        }
        return self._request("POST", "/sam3/visual_segment", body)

    def concept_segment(
        self,
        image_b64: str,
        prompts: list[dict[str, Any]],
        *,
        fmt: str = "binary",
        output_prob_thresh: float = 0.5,
    ) -> dict[str, Any]:
        body = {
            "image": {"type": "base64", "value": image_b64},
            "prompts": prompts,
            "format": fmt,
            "output_prob_thresh": output_prob_thresh,
        }
        return self._request("POST", "/sam3/concept_segment", body)


def image_file_to_b64(path: str | Path) -> tuple[str, Any]:
    """Return (base64, RGB uint8 array)."""
    import io

    import numpy as np
    from PIL import Image

    path = Path(path)
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    # re-encode so we know the payload matches the array we display
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, arr


def mask_from_prediction(pred: dict[str, Any]) -> "np.ndarray":
    """Convert a sidecar prediction (binary or polygon) to a bool HxW mask."""
    import numpy as np

    fmt = (pred.get("format") or "binary").lower()
    masks = pred.get("masks") or []
    if not masks:
        raise SidecarError("prediction has no masks")

    if fmt == "binary":
        m = np.asarray(masks[0])
        return m.astype(bool)

    if fmt in ("json", "polygon"):
        # polygons: list of rings of [x,y]
        try:
            import cv2
        except ImportError as e:
            raise SidecarError("opencv required to rasterize polygon masks") from e

        # need shape from bbox or caller — infer from points
        polys = masks
        max_x = max_y = 1
        for ring in polys:
            for x, y in ring:
                max_x = max(max_x, int(x) + 1)
                max_y = max(max_y, int(y) + 1)
        canvas = np.zeros((max_y, max_x), dtype=np.uint8)
        for ring in polys:
            pts = np.array(ring, dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(canvas, [pts], 1)
        return canvas.astype(bool)

    raise SidecarError(f"unsupported mask format: {fmt}")


def first_mask(response: dict[str, Any]) -> tuple["np.ndarray", float]:
    preds = response.get("predictions") or []
    if not preds:
        pr = response.get("prompt_results") or []
        if pr:
            preds = pr[0].get("predictions") or []
    if not preds:
        raise SidecarError("no predictions in response")
    pred = preds[0]
    conf = float(pred.get("confidence") or 0.0)
    return mask_from_prediction(pred), conf


# Avoid hard numpy import at module level for health-only scripts
def _np():
    import numpy as np

    return np
