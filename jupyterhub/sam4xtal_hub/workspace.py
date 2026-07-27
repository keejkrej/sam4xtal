"""Interactive SEM segmentation workspace for JupyterHub.

Runs against a local sidecar (see start_sidecar.sh). This is the supported
JupyterHub front-end: the Next.js app is optional and usually not reachable
through the Hub reverse proxy.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .client import SidecarClient, SidecarError, first_mask, image_file_to_b64

# Match web/src/lib/instance-colors.ts (export identity colors; never pure black).
INSTANCE_COLORS: list[tuple[int, int, int]] = [
    (34, 197, 94),
    (59, 130, 246),
    (245, 158, 11),
    (236, 72, 153),
    (20, 184, 166),
    (249, 115, 22),
    (14, 165, 233),
    (234, 179, 8),
    (168, 85, 247),
    (244, 63, 94),
    (132, 204, 22),
    (6, 182, 212),
]


def color_for_index(i: int) -> tuple[int, int, int]:
    return INSTANCE_COLORS[i % len(INSTANCE_COLORS)]


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def repo_root() -> Path:
    # .../sam4xtal/jupyterhub/sam4xtal_hub/workspace.py → repo root
    return Path(__file__).resolve().parent.parent.parent


def default_sidecar_url() -> str:
    return os.environ.get("SAM4XTAL_SIDECAR_URL") or os.environ.get(
        "INFERENCE_URL", "http://127.0.0.1:9001"
    )


def ensure_sidecar(
    url: str | None = None,
    *,
    mock: bool = False,
    start_if_missing: bool = True,
    wait_s: float = 600.0,
) -> SidecarClient:
    """Return a ready SidecarClient; optionally spawn start_sidecar.sh."""
    url = url or default_sidecar_url()
    client = SidecarClient(url)
    try:
        h = client.health()
        if h.get("load_state") != "loading":
            if h.get("load_state") == "error":
                raise SidecarError(f"sidecar error: {h.get('error')}", body=h)
            if h.get("ready") or h.get("backend") == "mock" or h.get("model_loaded"):
                return client
    except SidecarError:
        if not start_if_missing:
            raise

    if start_if_missing:
        script = repo_root() / "jupyterhub" / "start_sidecar.sh"
        if not script.is_file():
            raise SidecarError(f"missing {script}")
        args = ["bash", str(script)]
        if mock:
            args.append("--mock")
        print(f"[sam4xtal-hub] starting sidecar via {script} …")
        subprocess.run(args, check=False)
        # re-read env in case start script changed nothing for our process
        client = SidecarClient(os.environ.get("SAM4XTAL_SIDECAR_URL", url))

    print(f"[sam4xtal-hub] waiting for ready at {client.base_url} …")
    status = client.wait_ready(timeout_s=wait_s)
    print(f"[sam4xtal-hub] ready: backend={status.get('backend')} device={status.get('device')}")
    return client


@dataclass
class Instance:
    label: int
    points: list[dict[str, Any]] = field(default_factory=list)
    mask: Any = None  # np.ndarray bool or None
    confidence: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"Instance {self.label}"

    @property
    def color(self) -> tuple[int, int, int]:
        return color_for_index(self.label - 1)

    def measurement(self, nm_per_px: float | None = None) -> dict[str, Any]:
        import numpy as np

        if self.mask is None:
            return {}
        area = int(np.count_nonzero(self.mask))
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            bbox = [0, 0, 0, 0]
            bw = bh = 0
        else:
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            bbox = [x0, y0, x1, y1]
            bw, bh = x1 - x0, y1 - y0
        equiv = 2.0 * math.sqrt(area / math.pi) if area > 0 else 0.0
        out: dict[str, Any] = {
            "areaPx": area,
            "equivDiameterPx": equiv,
            "bboxWidthPx": bw,
            "bboxHeightPx": bh,
            "bbox_xyxy": bbox,
            "confidence": self.confidence,
        }
        if nm_per_px is not None and nm_per_px > 0:
            out["areaNm2"] = area * nm_per_px * nm_per_px
            out["equivDiameterNm"] = equiv * nm_per_px
            out["bboxWidthNm"] = bw * nm_per_px
            out["bboxHeightNm"] = bh * nm_per_px
        return out


@dataclass
class WorkspaceState:
    image_path: Path | None = None
    image_rgb: Any = None
    image_b64: str = ""
    instances: list[Instance] = field(default_factory=list)
    active: int = 0  # index into instances
    positive_mode: bool = True
    nm_per_px: float | None = None

    def ensure_instance(self) -> Instance:
        if not self.instances:
            self.instances.append(Instance(label=1))
            self.active = 0
        return self.instances[self.active]

    def new_instance(self) -> Instance:
        label = (max((i.label for i in self.instances), default=0)) + 1
        inst = Instance(label=label)
        self.instances.append(inst)
        self.active = len(self.instances) - 1
        return inst


def composite_overlay(rgb: Any, instances: list[Instance], alpha: float = 0.45) -> Any:
    import numpy as np

    out = rgb.astype(np.float32).copy()
    for inst in instances:
        if inst.mask is None:
            continue
        m = inst.mask.astype(bool)
        if m.shape[:2] != rgb.shape[:2]:
            continue
        color = np.array(inst.color, dtype=np.float32)
        out[m] = out[m] * (1 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def export_annotation(
    state: WorkspaceState,
    out_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write <stem>.mask.json + <stem>.mask.png (same scheme as the Next.js UI)."""
    import numpy as np
    from PIL import Image

    if state.image_path is None or state.image_rgb is None:
        raise SidecarError("no image loaded")
    out_dir = Path(out_dir) if out_dir else state.image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = state.image_path.name  # keep full filename as stem prefix
    json_path = out_dir / f"{stem}.mask.json"
    png_path = out_dir / f"{stem}.mask.png"

    h, w = state.image_rgb.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    instances_meta: list[dict[str, Any]] = []
    for inst in state.instances:
        if inst.mask is None:
            continue
        m = inst.mask.astype(bool)
        if m.shape != (h, w):
            continue
        rgb = inst.color
        canvas[m] = rgb
        meas = inst.measurement(state.nm_per_px)
        instances_meta.append(
            {
                "id": f"inst-{inst.label}",
                "label": inst.label,
                "name": inst.name,
                "color": {
                    "r": rgb[0],
                    "g": rgb[1],
                    "b": rgb[2],
                    "hex": rgb_to_hex(rgb),
                },
                "points": inst.points,
                "measurement": meas,
            }
        )

    Image.fromarray(canvas, mode="RGB").save(png_path)
    meta = {
        "maskEncoding": "instance-colors",
        "backgroundColor": {"r": 0, "g": 0, "b": 0, "hex": "#000000"},
        "sourceImage": state.image_path.name,
        "imageWidth": w,
        "imageHeight": h,
        "maskFileName": png_path.name,
        "nmPerPx": state.nm_per_px,
        "instances": instances_meta,
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return json_path, png_path


def load_image(state: WorkspaceState, path: str | Path) -> None:
    path = Path(path).expanduser().resolve()
    b64, arr = image_file_to_b64(path)
    state.image_path = path
    state.image_b64 = b64
    state.image_rgb = arr
    state.instances = [Instance(label=1)]
    state.active = 0


def segment_active(client: SidecarClient, state: WorkspaceState) -> Instance:
    inst = state.ensure_instance()
    if not inst.points:
        raise SidecarError("add at least one point before segmenting")
    if not state.image_b64:
        raise SidecarError("load an image first")
    resp = client.visual_segment(state.image_b64, inst.points, fmt="binary")
    mask, conf = first_mask(resp)
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    if state.image_rgb is not None and mask.shape != state.image_rgb.shape[:2]:
        # binary payload might be nested wrong — try squeeze
        mask = np.squeeze(mask).astype(bool)
    inst.mask = mask
    inst.confidence = conf
    return inst


def segment_all(client: SidecarClient, state: WorkspaceState) -> None:
    for i, inst in enumerate(state.instances):
        if not inst.points:
            continue
        state.active = i
        segment_active(client, state)


def launch_ui(
    client: SidecarClient | None = None,
    *,
    sidecar_url: str | None = None,
    mock: bool = False,
    sample_dir: str | Path | None = None,
):
    """Build an ipywidgets workspace. Call from a notebook cell.

    Returns (root_widget, state, client).
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "Need ipywidgets, matplotlib, numpy, Pillow in this kernel. "
            "Install into your Hub venv, then restart the kernel."
        ) from e

    if client is None:
        client = ensure_sidecar(sidecar_url, mock=mock, start_if_missing=True)

    state = WorkspaceState()
    if sample_dir is None:
        sample_dir = repo_root() / "samples" / "crystals"
    sample_dir = Path(sample_dir)

    status = widgets.HTML(value="<b>Status:</b> ready")
    path_in = widgets.Text(
        value=str(sample_dir / "fig02_A.png") if (sample_dir / "fig02_A.png").is_file() else "",
        description="Image:",
        layout=widgets.Layout(width="70%"),
        style={"description_width": "4rem"},
    )
    nm_in = widgets.FloatText(value=0.0, description="nm/px:", style={"description_width": "4rem"})
    mode = widgets.ToggleButtons(
        options=[("Positive click", True), ("Negative click", False)],
        value=True,
        description="Mode:",
    )
    out = widgets.Output()
    fig_holder: dict[str, Any] = {"fig": None, "ax": None}

    def set_status(msg: str, ok: bool = True) -> None:
        color = "#166534" if ok else "#991b1b"
        status.value = f"<b>Status:</b> <span style='color:{color}'>{msg}</span>"

    def redraw() -> None:
        with out:
            out.clear_output(wait=True)
            if state.image_rgb is None:
                print("Load an image to begin.")
                return
            fig, ax = plt.subplots(figsize=(8, 8))
            overlay = composite_overlay(state.image_rgb, state.instances)
            ax.imshow(overlay)
            for i, inst in enumerate(state.instances):
                for p in inst.points:
                    col = "lime" if p.get("positive", True) else "red"
                    marker = "o" if i == state.active else "x"
                    ax.plot(p["x"], p["y"], marker=marker, color=col, markersize=8)
                if i == state.active:
                    ax.set_title(
                        f"{state.image_path.name if state.image_path else ''} — "
                        f"active: {inst.name} ({len(inst.points)} pts)"
                    )
            ax.set_axis_off()
            fig.tight_layout()

            def on_click(event):
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    return
                if event.button != 1:
                    return
                inst = state.ensure_instance()
                inst.points.append(
                    {
                        "x": float(event.xdata),
                        "y": float(event.ydata),
                        "positive": bool(mode.value),
                    }
                )
                set_status(
                    f"point ({event.xdata:.0f}, {event.ydata:.0f}) "
                    f"{'pos' if mode.value else 'neg'} on {inst.name}"
                )
                redraw()

            fig.canvas.mpl_connect("button_press_event", on_click)
            fig_holder["fig"] = fig
            fig_holder["ax"] = ax
            plt.show()

    def on_load(_=None) -> None:
        try:
            p = path_in.value.strip()
            load_image(state, p)
            if nm_in.value and nm_in.value > 0:
                state.nm_per_px = float(nm_in.value)
            set_status(f"loaded {state.image_path} ({state.image_rgb.shape[1]}×{state.image_rgb.shape[0]})")
            redraw()
        except Exception as e:
            set_status(str(e), ok=False)

    def on_segment(_=None) -> None:
        try:
            if nm_in.value and nm_in.value > 0:
                state.nm_per_px = float(nm_in.value)
            inst = segment_active(client, state)
            m = inst.measurement(state.nm_per_px)
            set_status(
                f"segmented {inst.name}: area={m.get('areaPx')} px, "
                f"conf={inst.confidence:.3f}, t={client.base_url}"
            )
            redraw()
        except Exception as e:
            set_status(str(e), ok=False)

    def on_segment_all(_=None) -> None:
        try:
            segment_all(client, state)
            set_status(f"segmented {sum(1 for i in state.instances if i.mask is not None)} instance(s)")
            redraw()
        except Exception as e:
            set_status(str(e), ok=False)

    def on_new(_=None) -> None:
        inst = state.new_instance()
        set_status(f"new {inst.name}")
        redraw()

    def on_clear_pts(_=None) -> None:
        inst = state.ensure_instance()
        inst.points.clear()
        inst.mask = None
        set_status(f"cleared points on {inst.name}")
        redraw()

    def on_save(_=None) -> None:
        try:
            if nm_in.value and nm_in.value > 0:
                state.nm_per_px = float(nm_in.value)
            j, p = export_annotation(state)
            set_status(f"saved {j.name} + {p.name}")
        except Exception as e:
            set_status(str(e), ok=False)

    def on_health(_=None) -> None:
        try:
            h = client.health()
            set_status(f"health: {h}")
        except Exception as e:
            set_status(str(e), ok=False)

    btn_load = widgets.Button(description="Load", button_style="primary")
    btn_seg = widgets.Button(description="Segment active", button_style="success")
    btn_all = widgets.Button(description="Segment all")
    btn_new = widgets.Button(description="New instance")
    btn_clear = widgets.Button(description="Clear points")
    btn_save = widgets.Button(description="Save annotation", button_style="info")
    btn_health = widgets.Button(description="Health")

    btn_load.on_click(on_load)
    btn_seg.on_click(on_segment)
    btn_all.on_click(on_segment_all)
    btn_new.on_click(on_new)
    btn_clear.on_click(on_clear_pts)
    btn_save.on_click(on_save)
    btn_health.on_click(on_health)

    # Prefer interactive backend when available so clicks work.
    try:
        import matplotlib

        # ipympl is ideal; notebook backend is a fallback; inline often loses clicks
        for backend in ("widget", "ipympl", "notebook"):
            try:
                matplotlib.use(backend, force=False)
                break
            except Exception:
                continue
    except Exception:
        pass

    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            for magic in ("%matplotlib widget", "%matplotlib ipympl", "%matplotlib notebook"):
                try:
                    ip.run_line_magic("matplotlib", magic.split()[-1])
                    break
                except Exception:
                    continue
    except Exception:
        pass

    help_html = widgets.HTML(
        value=(
            "<p><b>sam4xtal JupyterHub workspace</b> — uses the local sidecar, "
            "not the Next.js UI. Click the image (needs <code>ipympl</code> / widget backend; "
            "if clicks do nothing, install <code>ipympl</code> and re-run this cell).</p>"
            f"<p>Sidecar: <code>{client.base_url}</code></p>"
        )
    )

    root = widgets.VBox(
        [
            help_html,
            status,
            widgets.HBox([path_in, btn_load]),
            widgets.HBox([nm_in, mode]),
            widgets.HBox([btn_seg, btn_all, btn_new, btn_clear, btn_save, btn_health]),
            out,
        ]
    )
    redraw()
    return root, state, client
