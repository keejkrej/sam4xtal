"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  CircleMinus,
  CirclePlus,
  Eraser,
  FolderOpen,
  Images,
  Loader2,
  Save,
  Sparkles,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { ImageCanvas } from "@/components/workspace/image-canvas";
import {
  buildAnnotationDownload,
  extractPolygons,
  measureCrystal,
  runVisualSegment,
} from "@/lib/sam3";
import type { WorkspaceImage } from "@/lib/types";
import { formatNumber } from "@/lib/utils";
import { getCachedImage } from "@/lib/image-cache";
import { EMPTY_WORK, useWorkspaceStore } from "@/stores/workspace";

function stripDataUrl(dataUrl: string): string {
  const idx = dataUrl.indexOf(",");
  return idx >= 0 ? dataUrl.slice(idx + 1) : dataUrl;
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function imageDims(dataUrl: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error("Failed to decode image"));
    img.src = dataUrl;
  });
}

async function loadWorkspaceImage(file: File): Promise<WorkspaceImage> {
  const id = `${file.name}-${file.size}`;
  const cached = await getCachedImage(id);
  const dataUrl = cached ?? (await readFileAsDataUrl(file));
  const dims = await imageDims(dataUrl);
  return {
    id,
    name: file.name,
    width: dims.width,
    height: dims.height,
    mimeType: file.type || "image/png",
    dataUrl,
  };
}

export function Workspace() {
  const hasHydrated = useWorkspaceStore((s) => s.hasHydrated);
  const images = useWorkspaceStore((s) => s.images);
  const index = useWorkspaceStore((s) => s.index);
  const nmPerPx = useWorkspaceStore((s) => s.nmPerPx);
  const negativeMode = useWorkspaceStore((s) => s.negativeMode);
  const setImages = useWorkspaceStore((s) => s.setImages);
  const goTo = useWorkspaceStore((s) => s.goTo);
  const setNmPerPx = useWorkspaceStore((s) => s.setNmPerPx);
  const setNegativeMode = useWorkspaceStore((s) => s.setNegativeMode);
  const addPoint = useWorkspaceStore((s) => s.addPoint);
  const setWork = useWorkspaceStore((s) => s.setWork);
  const clearCurrentWork = useWorkspaceStore((s) => s.clearCurrentWork);
  const upsertSaved = useWorkspaceStore((s) => s.upsertSaved);
  const current = useWorkspaceStore((s) => s.images[s.index] ?? null);
  const work = useWorkspaceStore((s) => {
    const img = s.images[s.index];
    if (!img) return EMPTY_WORK;
    return s.workByImageId[img.id] ?? EMPTY_WORK;
  });

  const [busy, setBusy] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const unsub = useWorkspaceStore.persist.onFinishHydration(() => {
      useWorkspaceStore.getState().setHasHydrated(true);
    });
    if (useWorkspaceStore.persist.hasHydrated()) {
      useWorkspaceStore.getState().setHasHydrated(true);
    }
    return unsub;
  }, []);

  const parsedNmPerPx = useMemo(() => {
    const n = Number(nmPerPx);
    return Number.isFinite(n) && n > 0 ? n : null;
  }, [nmPerPx]);

  const measurement = useMemo(() => {
    if (!work.prediction) return null;
    return measureCrystal(work.prediction, parsedNmPerPx);
  }, [work.prediction, parsedNmPerPx]);

  async function loadImageFiles(imageFiles: File[], label = "images") {
    if (!imageFiles.length) {
      toast.error("No image files found.");
      return;
    }
    imageFiles.sort((a, b) => a.name.localeCompare(b.name));
    const loadingId = toast.loading(`Loading ${label}…`);
    try {
      const loaded: WorkspaceImage[] = [];
      for (const file of imageFiles) {
        loaded.push(await loadWorkspaceImage(file));
      }
      setImages(loaded);
      toast.success(`Loaded ${loaded.length} image${loaded.length === 1 ? "" : "s"}`, {
        id: loadingId,
      });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to load images", {
        id: loadingId,
      });
    }
  }

  async function onFilesSelected(files: FileList | null) {
    if (!files?.length) return;
    const imageFiles = Array.from(files).filter((f) =>
      /\.(png|jpe?g|tif?f|bmp|webp)$/i.test(f.name),
    );
    await loadImageFiles(imageFiles);
  }

  async function loadSampleFiles() {
    const loadingId = toast.loading("Loading samples…");
    try {
      const listRes = await fetch("/api/samples/crystals");
      if (!listRes.ok) {
        throw new Error("Could not list sample files");
      }
      const { files } = (await listRes.json()) as { files?: string[] };
      if (!files?.length) {
        toast.error("No sample files in samples/crystals", { id: loadingId });
        return;
      }
      const imageFiles: File[] = [];
      for (const name of files) {
        const res = await fetch(`/api/samples/crystals/${encodeURIComponent(name)}`);
        if (!res.ok) {
          throw new Error(`Failed to fetch ${name}`);
        }
        const blob = await res.blob();
        imageFiles.push(
          new File([blob], name, { type: blob.type || "image/png" }),
        );
      }
      toast.dismiss(loadingId);
      await loadImageFiles(imageFiles, "samples");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to load samples", {
        id: loadingId,
      });
    }
  }

  async function segment() {
    if (!current || work.points.length === 0) return;
    setBusy(true);
    const loadingId = toast.loading("Running segmentation…");
    try {
      const res = await runVisualSegment({
        image: { type: "base64", value: stripDataUrl(current.dataUrl) },
        points: work.points,
        onStatus: (msg) => toast.loading(msg, { id: loadingId }),
      });
      const pred =
        res.predictions?.[0] ??
        res.prompt_results?.[0]?.predictions?.[0] ??
        null;
      if (!pred) {
        throw new Error("No mask returned from SAM3");
      }
      setWork({
        prediction: pred,
        polygons: extractPolygons(pred),
      });
      toast.success(`Segmented in ${formatNumber(res.time * 1000, 0)} ms`, {
        id: loadingId,
      });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Segmentation failed", {
        id: loadingId,
      });
    } finally {
      setBusy(false);
    }
  }

  async function saveAnnotation() {
    if (!current || !work.prediction || !measurement || work.polygons.length === 0) {
      return;
    }
    setBusy(true);
    try {
      const { metaFileName, maskFileName, metaJson, maskPng, meta } =
        await buildAnnotationDownload({
          imageId: current.id,
          imageName: current.name,
          width: current.width,
          height: current.height,
          points: work.points,
          polygons: work.polygons,
          measurement,
          nmPerPx: parsedNmPerPx,
          bbox_xyxy: work.prediction.bbox_xyxy,
        });
      upsertSaved(meta);

      const downloadBlob = (blob: Blob, name: string) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = name;
        a.click();
        URL.revokeObjectURL(url);
      };
      downloadBlob(new Blob([metaJson], { type: "application/json" }), metaFileName);
      downloadBlob(maskPng, maskFileName);
      toast.success(`Saved ${metaFileName} + ${maskFileName}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save annotation");
    } finally {
      setBusy(false);
    }
  }

  if (!hasHydrated) {
    return (
      <div className="flex min-h-0 flex-1 items-center justify-center p-4 text-sm text-muted-foreground">
        Restoring session…
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 p-4 md:p-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">
          sam4xtal
        </h1>
      </header>

      <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[280px_1fr]">
        <Card className="min-h-0 overflow-y-auto">
          <CardContent className="flex flex-col gap-4">
            <div className="space-y-2">
              <Label>File</Label>
              <div className="flex flex-col gap-1.5">
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept="image/*"
                  className="sr-only"
                  tabIndex={-1}
                  aria-hidden="true"
                  onChange={(e) => {
                    onFilesSelected(e.target.files);
                    e.target.value = "";
                  }}
                />
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <FolderOpen />
                  {images.length ? "Replace files" : "Choose files"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={loadSampleFiles}
                  disabled={busy}
                >
                  <Images />
                  Load samples
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="nm">Resolution</Label>
              <Input
                id="nm"
                type="number"
                min={0}
                step="any"
                placeholder="e.g. 2.5"
                value={nmPerPx}
                onChange={(e) => setNmPerPx(e.target.value)}
              />
            </div>

            <Separator />

            <div className="space-y-2">
              <Label>Click</Label>
              <div className="flex flex-col gap-2">
                <Button
                  type="button"
                  variant={negativeMode ? "outline" : "default"}
                  size="sm"
                  onClick={() => setNegativeMode(false)}
                >
                  <CirclePlus />
                  Positive
                </Button>
                <Button
                  type="button"
                  variant={negativeMode ? "default" : "outline"}
                  size="sm"
                  onClick={() => setNegativeMode(true)}
                >
                  <CircleMinus />
                  Negative
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={clearCurrentWork}
                  disabled={!work.points.length && !work.polygons.length}
                >
                  <Eraser />
                  Clear
                </Button>
              </div>
            </div>

            <Separator />

            <Button
              type="button"
              onClick={segment}
              disabled={!current || work.points.length === 0 || busy}
            >
              {busy ? <Loader2 className="animate-spin" /> : <Sparkles />}
              Run segmentation
            </Button>

            <Button
              type="button"
              variant="secondary"
              onClick={saveAnnotation}
              disabled={!work.prediction || !measurement || work.polygons.length === 0 || busy}
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
                    className="flex-1"
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
                    className="flex-1"
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
                <p className="font-medium">Size</p>
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
          </CardContent>
        </Card>

        <Card className="flex h-full min-h-0 flex-col overflow-hidden py-0">
          <CardContent className="min-h-0 flex-1 p-0">
            <div className="h-full min-h-0">
              {current ? (
                <ImageCanvas
                  src={current.dataUrl}
                  points={work.points}
                  polygons={work.polygons}
                  negativeMode={negativeMode}
                  disabled={busy}
                  onAddPoint={addPoint}
                />
              ) : (
                <div className="flex h-full min-h-[320px] flex-col items-center justify-center gap-3 bg-muted/30 p-8 text-center lg:min-h-0">
                  <FolderOpen className="size-10 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    Select image files to begin (PNG, JPEG, TIFF, WEBP).
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
