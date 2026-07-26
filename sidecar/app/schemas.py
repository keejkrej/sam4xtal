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
    """Top-left anchored XYWH for PCS concept_segment."""

    x: float
    y: float
    width: float
    height: float


class BoxXYXY(BaseModel):
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
            return [p if isinstance(p, VisualPrompt) else VisualPrompt.model_validate(p) for p in raw]
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


class Sam3Prompt(BaseModel):
    type: Literal["text", "visual"] = "text"
    text: Optional[str] = None
    boxes: Optional[list[Union[BoxTopLeftXYWH, BoxXYXY, dict[str, Any]]]] = None
    box_labels: Optional[list[Union[int, bool]]] = None
    output_prob_thresh: Optional[float] = None


class ConceptSegmentRequest(BaseModel):
    image: InferenceRequestImage
    prompts: list[Sam3Prompt] = Field(min_length=1)
    format: Literal["polygon", "rle", "json"] = "polygon"
    image_id: Optional[str] = None
    output_prob_thresh: float = 0.5
    model_id: str = "sam3/sam3_final"
    nms_iou_threshold: Optional[float] = None


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
