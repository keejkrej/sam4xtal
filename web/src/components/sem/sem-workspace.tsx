"use client";

import { useMemo, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Eraser,
  FolderOpen,
  Loader2,
  MousePointer2,
  Save,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ImageCanvas } from "@/components/sem/image-canvas";
import {
  extractPolygons,
  fileToBase64Image,
  measureCrystal,
  runVisualSegment,
} from "@/lib/sam3";
import type {
  AnnotationResult,
  PointPrompt,
  SemImage,
  SegmentationPrediction,
} from "@/lib/types";
import { formatNumber } from "@/lib/utils";

async function loadImageMeta(file: File, objectUrl: string): Promise<SemImage> {
  const dims = await new Promise<{ width: number; height: number }>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error(`Failed to load ${file.name}`));
    img.src = objectUrl;
  });
  return {
    id: `${file.name}-${file.size}-${file.lastModified}`,
    name: file.name,
    file,
    objectUrl,
    width: dims.width,
    height: dims.height,
  };
}

export function SemWorkspace() {
  const [images, setImages] = useState<SemImage[]>([]);
  const [index, setIndex] = useState(0);
  const [points, setPoints] = useState<PointPrompt[]>([]);
  const [polygons, setPolygons] = useState<number[][][]>([]);
  const [prediction, setPrediction] = useState<SegmentationPrediction | null>(null);
  const [nmPerPx, setNmPerPx] = useState<string>("");
  const [negativeMode, setNegativeMode] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<AnnotationResult[]>([]);
  const [status, setStatus] = useState<string | null>(null);

  const current = images[index] ?? null;
  const parsedNmPerPx = useMemo(() => {
    const n = Number(nmPerPx);
    return Number.isFinite(n) && n > 0 ? n : null;
  }, [nmPerPx]);

  const measurement = useMemo(() => {
    if (!prediction) return null;
    return measureCrystal(prediction, parsedNmPerPx);
  }, [prediction, parsedNmPerPx]);

  async function onFolderSelected(files: FileList | null) {
    if (!files?.length) return;
    setError(null);
    const imageFiles = Array.from(files).filter((f) =>
      /\.(png|jpe?g|tif?f|bmp|webp)$/i.test(f.name),
    );
    if (!imageFiles.length) {
      setError("No SEM image files found in that folder.");
      return;
    }
    imageFiles.sort((a, b) => a.name.localeCompare(b.name));
    const loaded: SemImage[] = [];
    for (const file of imageFiles) {
      const url = URL.createObjectURL(file);
      loaded.push(await loadImageMeta(file, url));
    }
    // revoke previous
    images.forEach((img) => URL.revokeObjectURL(img.objectUrl));
    setImages(loaded);
    setIndex(0);
    resetAnnotation();
    setStatus(`Loaded ${loaded.length} image${loaded.length === 1 ? "" : "s"}`);
  }

  function resetAnnotation() {
    setPoints([]);
    setPolygons([]);
    setPrediction(null);
    setError(null);
  }

  function goTo(next: number) {
    if (!images.length) return;
    const clamped = Math.max(0, Math.min(images.length - 1, next));
    setIndex(clamped);
    resetAnnotation();
  }

  async function segment() {
    if (!current || points.length === 0) return;
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      const image = await fileToBase64Image(current.file);
      const res = await runVisualSegment({ image, points });
      const pred =
        res.predictions?.[0] ??
        res.prompt_results?.[0]?.predictions?.[0] ??
        null;
      if (!pred) {
        throw new Error("No mask returned from SAM3");
      }
      setPrediction(pred);
      setPolygons(extractPolygons(pred));
      setStatus(`Segmented in ${formatNumber(res.time * 1000, 0)} ms`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Segmentation failed");
    } finally {
      setBusy(false);
    }
  }

  function saveAnnotation() {
    if (!current || !prediction || !measurement) return;
    const result: AnnotationResult = {
      imageId: current.id,
      imageName: current.name,
      points,
      maskPolygons: polygons,
      measurement,
      nmPerPx: parsedNmPerPx,
      savedAt: new Date().toISOString(),
    };
    setSaved((prev) => {
      const without = prev.filter((r) => r.imageId !== current.id);
      return [...without, result];
    });

    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${current.name.replace(/\.[^.]+$/, "")}.mask.json`;
    a.click();
    URL.revokeObjectURL(url);
    setStatus(`Saved ${a.download}`);
  }

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-4 p-4 md:p-6">
      <header className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-sm font-medium tracking-wide text-muted-foreground">
            sam4xtal
          </p>
          <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">
            SEM crystal segmentation
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Load a folder of SEM images, click the crystal as point prompts, run
            SAM3, and save mask + size (px / nm).
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="secondary">Roboflow-compatible API</Badge>
          <Badge variant="outline">/sam3/visual_segment</Badge>
        </div>
      </header>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Controls</CardTitle>
            <CardDescription>Folder, resolution, prompts</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="space-y-2">
              <Label htmlFor="folder">SEM folder</Label>
              <Input
                id="folder"
                type="file"
                multiple
                accept="image/*"
                ref={(el) => {
                  if (el) {
                    el.setAttribute("webkitdirectory", "");
                    el.setAttribute("directory", "");
                  }
                }}
                onChange={(e) => onFolderSelected(e.target.files)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="nm">SEM resolution (nm / px)</Label>
              <Input
                id="nm"
                type="number"
                min={0}
                step="any"
                placeholder="e.g. 2.5"
                value={nmPerPx}
                onChange={(e) => setNmPerPx(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Optional. Converts mask area and equivalent diameter to nm.
              </p>
            </div>

            <Separator />

            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant={negativeMode ? "outline" : "default"}
                size="sm"
                onClick={() => setNegativeMode(false)}
              >
                <MousePointer2 />
                Positive
              </Button>
              <Button
                type="button"
                variant={negativeMode ? "default" : "outline"}
                size="sm"
                onClick={() => setNegativeMode(true)}
              >
                Negative
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={resetAnnotation}
                disabled={!points.length && !polygons.length}
              >
                <Eraser />
                Clear
              </Button>
            </div>

            <Button
              type="button"
              onClick={segment}
              disabled={!current || points.length === 0 || busy}
            >
              {busy ? <Loader2 className="animate-spin" /> : <Sparkles />}
              Run segmentation
            </Button>

            <Button
              type="button"
              variant="secondary"
              onClick={saveAnnotation}
              disabled={!prediction || !measurement}
            >
              <Save />
              Save annotation
            </Button>

            {current && (
              <div className="rounded-lg border bg-muted/40 p-3 text-sm">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="truncate font-medium">{current.name}</span>
                  <span className="shrink-0 text-muted-foreground">
                    {index + 1}/{images.length}
                  </span>
                </div>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => goTo(index - 1)}
                    disabled={index <= 0}
                  >
                    <ChevronLeft />
                    Prev
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => goTo(index + 1)}
                    disabled={index >= images.length - 1}
                  >
                    Next
                    <ChevronRight />
                  </Button>
                </div>
              </div>
            )}

            {measurement && (
              <div className="space-y-1 rounded-lg border p-3 text-sm">
                <p className="font-medium">Crystal size</p>
                <p>
                  Area:{" "}
                  <span className="font-mono">{formatNumber(measurement.areaPx, 0)}</span>{" "}
                  px²
                </p>
                <p>
                  Eq. diameter:{" "}
                  <span className="font-mono">
                    {formatNumber(measurement.equivDiameterPx, 1)}
                  </span>{" "}
                  px
                </p>
                {measurement.areaNm2 != null && (
                  <>
                    <p>
                      Area:{" "}
                      <span className="font-mono">
                        {formatNumber(measurement.areaNm2, 1)}
                      </span>{" "}
                      nm²
                    </p>
                    <p>
                      Eq. diameter:{" "}
                      <span className="font-mono">
                        {formatNumber(measurement.equivDiameterNm ?? 0, 1)}
                      </span>{" "}
                      nm
                    </p>
                  </>
                )}
                <p className="text-muted-foreground">
                  Confidence {formatNumber(measurement.confidence * 100, 1)}%
                </p>
              </div>
            )}

            {saved.length > 0 && (
              <p className="text-xs text-muted-foreground">
                {saved.length} annotation{saved.length === 1 ? "" : "s"} saved this
                session
              </p>
            )}

            {status && <p className="text-xs text-muted-foreground">{status}</p>}
            {error && (
              <p className="whitespace-pre-wrap text-xs text-destructive">{error}</p>
            )}
          </CardContent>
        </Card>

        <Card className="overflow-hidden">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">
              {current ? current.name : "No image loaded"}
            </CardTitle>
            <CardDescription>
              {current
                ? `${current.width}×${current.height} · click crystal to add point prompts`
                : "Choose a folder of SEM images to begin"}
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-[min(70vh,720px)] border-t">
              {current ? (
                <ImageCanvas
                  src={current.objectUrl}
                  points={points}
                  polygons={polygons}
                  negativeMode={negativeMode}
                  disabled={busy}
                  onAddPoint={(p) => setPoints((prev) => [...prev, p])}
                />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-3 bg-muted/30 p-8 text-center">
                  <FolderOpen className="size-10 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    Select a folder containing SEM micrographs (PNG, JPEG, TIFF,
                    WEBP).
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
