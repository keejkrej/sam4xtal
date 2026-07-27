"""FastAPI SAM3 sidecar — API-compatible with Roboflow Inference.

Endpoints mirror https://inference.roboflow.com/foundation/sam3/ :

  POST /sam3/embed_image
  POST /sam3/visual_segment     # PVS: point/box → one object
  POST /sam3/concept_segment    # PCS: text and/or exemplar boxes → all matches

Swap this sidecar for Roboflow by pointing the Next.js proxy at
https://serverless.roboflow.com (or a hosted inference server) and setting
ROBOFLOW_API_KEY.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
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

# Default to real SAM unless the launcher passed --mock.
os.environ.setdefault("SAM3_BACKEND", "transformers")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Non-blocking: accept health checks while weights download.
    # Preload Tracker (PVS clicks) and Sam3Model (concept transfer / PCS)
    # so first Transfer in the UI is not a long cold start.
    status = engine.start_backend_load()
    concept = engine.start_concept_model_load()
    print(
        f"[sam4xtal] starting backend={status['backend']} "
        f"load_state={status['load_state']} "
        f"concept_load_state={concept.get('concept_load_state')} "
        f"model_id={status['model_id']}",
        flush=True,
    )
    yield


app = FastAPI(
    title="sam4xtal SAM3 Sidecar",
    version="0.1.0",
    description="Roboflow-compatible local SAM3 inference sidecar",
    lifespan=lifespan,
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
    status = engine.backend_status()
    return {
        "ok": True,
        "service": "sam4xtal-sam3-sidecar",
        **status,
    }


def _raise_if_not_ready() -> None:
    try:
        engine.ensure_ready()
    except engine.ModelNotReady as exc:
        status = 503 if exc.state == "loading" else 500
        raise HTTPException(
            status_code=status,
            detail={
                "message": str(exc),
                "load_state": exc.state,
                **engine.backend_status(),
            },
        ) from exc


@app.post("/sam3/embed_image", response_model=EmbedImageResponse)
def sam3_embed_image(
    body: EmbedImageRequest,
    api_key: Optional[str] = Query(default=None),
) -> EmbedImageResponse:
    _ = api_key  # accepted for Roboflow API compatibility
    _raise_if_not_ready()
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
    _raise_if_not_ready()
    try:
        return engine.visual_segment(body)
    except engine.ModelNotReady as exc:
        status = 503 if exc.state == "loading" else 500
        raise HTTPException(
            status_code=status,
            detail={
                "message": str(exc),
                "load_state": exc.state,
                **engine.backend_status(),
            },
        ) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/sam3/concept_segment", response_model=SegmentationResponse)
def sam3_concept_segment(
    body: ConceptSegmentRequest,
    api_key: Optional[str] = Query(default=None),
) -> SegmentationResponse:
    """PCS: text and/or image exemplars (Roboflow-compatible).

    Few-shot in-image: pass corrected-instance bboxes as visual exemplars
    (``type: visual``, ``boxes``, ``box_labels``) to find all matches.
    """
    _ = api_key
    try:
        return engine.concept_segment(body)
    except engine.ModelNotReady as exc:
        status = 503 if exc.state == "loading" else 500
        raise HTTPException(
            status_code=status,
            detail={
                "message": str(exc),
                "load_state": exc.state,
                **engine.backend_status(),
            },
        ) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
