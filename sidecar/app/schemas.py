"""Request/response schemas aligned with Roboflow Inference SAM3 HTTP API.

See: https://inference.roboflow.com/foundation/sam3/
     https://docs.roboflow.com/deploy/supported-models/sam3
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class InferenceRequestImage(BaseModel):
    type: Literal["url", "base64"]
    value: str


class Point(BaseModel):
    x: float
    y: float
    positive: bool = True


class BoxXYWH(BaseModel):
    """Center-anchored XYWH for PVS visual_segment (Roboflow convention)."""

    x: float
    y: float
    width: float
    height: float


class BoxTopLeftXYWH(BaseModel):
    """Top-left anchored XYWH for PCS concept_segment (Roboflow convention)."""

    x: float
    y: float
    width: float
    height: float


class BoxXYXY(BaseModel):
    """Corner form accepted by Roboflow concept_segment boxes."""

    x0: float
    y0: float
    x1: float
    y1: float


class VisualPrompt(BaseModel):
    points: Optional[list[Point]] = None
    box: Optional[BoxXYWH] = None

    @model_validator(mode="after")
    def require_points_or_box(self) -> "VisualPrompt":
        if not self.points and self.box is None:
            raise ValueError("Each visual prompt must include points and/or a box")
        return self


class Sam2PromptSet(BaseModel):
    """Roboflow wraps prompts either as a list or as `{prompts: [...]}`."""

    prompts: list[VisualPrompt]


class VisualSegmentRequest(BaseModel):
    image: Optional[InferenceRequestImage] = None
    image_id: Optional[str] = None
    prompts: Optional[Union[list[VisualPrompt], Sam2PromptSet, dict[str, Any]]] = None
    format: Literal["json", "rle", "binary", "polygon"] = "json"
    multimask_output: bool = True
    save_logits_to_cache: bool = False
    load_logits_from_cache: bool = False
    model_id: str = "sam3/sam3_final"

    def normalized_prompts(self) -> list[VisualPrompt]:
        raw = self.prompts
        if raw is None:
            return []
        if isinstance(raw, Sam2PromptSet):
            return raw.prompts
        if isinstance(raw, list):
            return [
                p if isinstance(p, VisualPrompt) else VisualPrompt.model_validate(p)
                for p in raw
            ]
        if isinstance(raw, dict):
            if "prompts" in raw:
                return [VisualPrompt.model_validate(p) for p in raw["prompts"]]
            return [VisualPrompt.model_validate(raw)]
        return []


class EmbedImageRequest(BaseModel):
    image: InferenceRequestImage
    image_id: Optional[str] = None
    model_id: str = "sam3/sam3_final"


class EmbedImageResponse(BaseModel):
    image_id: str
    image_shape: list[int]
    time: float


def box_to_xyxy(box: Union[BoxTopLeftXYWH, BoxXYXY, dict[str, Any]]) -> list[float]:
    """Normalize a Roboflow PCS box to absolute-pixel xyxy."""
    if isinstance(box, BoxXYXY):
        return [float(box.x0), float(box.y0), float(box.x1), float(box.y1)]
    if isinstance(box, BoxTopLeftXYWH):
        return [
            float(box.x),
            float(box.y),
            float(box.x + box.width),
            float(box.y + box.height),
        ]
    if isinstance(box, dict):
        if all(k in box for k in ("x0", "y0", "x1", "y1")):
            return [
                float(box["x0"]),
                float(box["y0"]),
                float(box["x1"]),
                float(box["y1"]),
            ]
        if all(k in box for k in ("x", "y", "width", "height")):
            x, y = float(box["x"]), float(box["y"])
            return [x, y, x + float(box["width"]), y + float(box["height"])]
    raise ValueError(
        "PCS box must be {x0,y0,x1,y1} or top-left {x,y,width,height}"
    )


def box_label_to_int(lab: Union[int, bool]) -> int:
    """Roboflow: 1 = positive exemplar, 0 = negative exemplar."""
    if isinstance(lab, bool):
        return 1 if lab else 0
    return 0 if int(lab) == 0 else 1


class Sam3Prompt(BaseModel):
    """One concept prompt for Roboflow ``/sam3/concept_segment``.

    - ``type="text"``: open-vocab noun phrase (``text``)
    - ``type="visual"``: image exemplars — box one/more example objects;
      SAM3 finds every similar instance on the same image (few-shot PCS).
      ``box_labels`` is required with ``boxes``: 1 positive, 0 negative.
    - Combined: ``text`` + ``boxes`` + ``box_labels`` in one prompt.
    """

    type: Literal["text", "visual"] = "text"
    text: Optional[str] = None
    boxes: Optional[list[Union[BoxTopLeftXYWH, BoxXYXY, dict[str, Any]]]] = None
    box_labels: Optional[list[Union[int, bool]]] = None
    output_prob_thresh: Optional[float] = None

    def boxes_xyxy(self) -> list[list[float]]:
        return [box_to_xyxy(b) for b in (self.boxes or [])]

    def labels_int(self) -> list[int]:
        boxes = self.boxes or []
        labels = self.box_labels
        if labels is None:
            # Default all positive when omitted (local convenience); Roboflow
            # docs say box_labels is required with boxes.
            return [1] * len(boxes)
        if len(labels) != len(boxes):
            raise ValueError("box_labels length must match boxes length")
        return [box_label_to_int(lab) for lab in labels]

    @model_validator(mode="after")
    def validate_prompt_content(self) -> "Sam3Prompt":
        has_text = bool(self.text and str(self.text).strip())
        has_boxes = bool(self.boxes)
        if has_boxes and self.box_labels is not None:
            if len(self.box_labels) != len(self.boxes or []):
                raise ValueError("box_labels length must match boxes length")
        if self.type == "text" and not has_text and not has_boxes:
            raise ValueError('text prompt requires "text" and/or exemplar boxes')
        if self.type == "visual" and not has_boxes and not has_text:
            raise ValueError(
                'visual prompt requires exemplar "boxes" (+ box_labels) '
                "and/or text"
            )
        if has_boxes and self.box_labels is None:
            # Roboflow requires box_labels; we default to all-positive but
            # callers should send them explicitly for strict compatibility.
            pass
        return self


class ConceptSegmentRequest(BaseModel):
    """Roboflow-compatible PCS request (text and/or image exemplars).

    Few-shot “correct a few masks → find the rest” maps to one visual prompt
    whose ``boxes`` are the seed instance bboxes and ``box_labels`` are 1/0.
    """

    # Roboflow sends image; image_id is a local cache convenience
    image: Optional[InferenceRequestImage] = None
    image_id: Optional[str] = None
    prompts: list[Sam3Prompt] = Field(min_length=1)
    format: Literal["polygon", "rle", "json"] = "polygon"
    output_prob_thresh: float = 0.5
    model_id: str = "sam3/sam3_final"
    nms_iou_threshold: Optional[float] = None

    @model_validator(mode="after")
    def require_image_or_id(self) -> "ConceptSegmentRequest":
        if self.image is None and not self.image_id:
            raise ValueError("Either image or image_id is required")
        return self


class SegmentationPrediction(BaseModel):
    format: str
    confidence: float
    masks: list[Any]
    # Optional extras used by the UI (not required by Roboflow)
    area_px: Optional[int] = None
    bbox_xyxy: Optional[list[float]] = None


class PromptResult(BaseModel):
    prompt_index: int
    predictions: list[SegmentationPrediction]
    echo: Optional[dict[str, Any]] = None


class SegmentationResponse(BaseModel):
    """Compatible with both Inference HTTP and serverless response shapes."""

    time: float
    prompt_results: list[PromptResult]
    # Convenience mirror of first prompt's predictions (serverless-style)
    predictions: list[SegmentationPrediction] = Field(default_factory=list)
