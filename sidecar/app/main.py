"""FastAPI SAM3 sidecar — API-compatible with Roboflow Inference.

Endpoints mirror https://inference.roboflow.com/foundation/sam3/ :

  POST /sam3/embed_image
  POST /sam3/visual_segment
  POST /sam3/concept_segment

Swap this sidecar for Roboflow by pointing the Next.js proxy at
https://serverless.roboflow.com (or a hosted inference server) and setting
ROBOFLOW_API_KEY.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import engine
from .schemas import (
    ConceptSegmentRequest,
    EmbedImageRequest,
    EmbedImageResponse,
    SegmentationResponse,
    VisualSegmentRequest,
)

app = FastAPI(
    title="sam4xtal SAM3 Sidecar",
    version="0.1.0",
    description="Roboflow-compatible local SAM3 inference sidecar",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "backend": os.environ.get("SAM3_BACKEND", "mock"),
        "service": "sam4xtal-sam3-sidecar",
    }


@app.post("/sam3/embed_image", response_model=EmbedImageResponse)
def sam3_embed_image(
    body: EmbedImageRequest,
    api_key: Optional[str] = Query(default=None),
) -> EmbedImageResponse:
    _ = api_key  # accepted for Roboflow API compatibility
    try:
        return engine.embed_image(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/sam3/visual_segment", response_model=SegmentationResponse)
def sam3_visual_segment(
    body: VisualSegmentRequest,
    api_key: Optional[str] = Query(default=None),
) -> SegmentationResponse:
    _ = api_key
    try:
        return engine.visual_segment(body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/sam3/concept_segment", response_model=SegmentationResponse)
def sam3_concept_segment(
    body: ConceptSegmentRequest,
    api_key: Optional[str] = Query(default=None),
) -> SegmentationResponse:
    _ = api_key
    try:
        return engine.concept_segment(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
