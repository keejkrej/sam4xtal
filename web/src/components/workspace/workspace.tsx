"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleMinus,
  CirclePlus,
  Eraser,
  FolderOpen,
  Images,
  Loader2,
  MousePointer2,
  Plus,
  Save,
  Sparkles,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { ThemeToggle } from "@/components/theme-toggle";
import { ImageCanvas } from "@/components/workspace/image-canvas";
import { colorForInstance } from "@/lib/instance-colors";
import {
  boxIoU,
  buildAnnotationDownload,
  extractPolygons,
  instanceBBoxXyxy,
  instancesReadyToSave,
  measureCrystal,
  predictionBBoxXyxy,
  runConceptSegment,
  runVisualSegment,
} from "@/lib/sam3";
import type { SegmentInstance, WorkspaceImage } from "@/lib/types";
import { createEmptyInstance } from "@/stores/workspace";
import {
  formatNmPerPx,
  readSemResolutionFromFile,
  resolutionSourceLabel,
} from "@/lib/sem-resolution";
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
    img.onload = () =>
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error("Failed to decode image"));
    img.src = dataUrl;
  });
}

async function loadWorkspaceImage(file: File): Promise<WorkspaceImage> {
  const id = `${file.name}-${file.size}`;
  const cached = await getCachedImage(id);
  const dataUrl = cached ?? (await readFileAsDataUrl(file));
  const dims = await imageDims(dataUrl);
  const resolution = await readSemResolutionFromFile(file);
  return {
    id,
    name: file.name,
    width: dims.width,
    height: dims.height,
    mimeType: file.type || "image/png",
    dataUrl,
    nmPerPx: resolution ? formatNmPerPx(resolution.nmPerPx) : "",
    nmPerPxFromFile: resolution?.nmPerPx ?? null,
    nmPerPxSource: resolution?.source ?? null,
  };
}

export function Workspace() {
  const hasHydrated = useWorkspaceStore((s) => s.hasHydrated);
  const images = useWorkspaceStore((s) => s.images);
  const index = useWorkspaceStore((s) => s.index);
  const clickMode = useWorkspaceStore((s) => s.clickMode);
  const setImages = useWorkspaceStore((s) => s.setImages);
  const goTo = useWorkspaceStore((s) => s.goTo);
  const setNmPerPx = useWorkspaceStore((s) => s.setNmPerPx);
  const setClickMode = useWorkspaceStore((s) => s.setClickMode);
  const addPoint = useWorkspaceStore((s) => s.addPoint);
  const removePoint = useWorkspaceStore((s) => s.removePoint);
  const addInstance = useWorkspaceStore((s) => s.addInstance);
  const selectInstance = useWorkspaceStore((s) => s.selectInstance);
  const removeInstance = useWorkspaceStore((s) => s.removeInstance);
  const updateActiveInstance = useWorkspaceStore((s) => s.updateActiveInstance);
  const updateInstance = useWorkspaceStore((s) => s.updateInstance);
  const setInstances = useWorkspaceStore((s) => s.setInstances);
  const clearActiveInstance = useWorkspaceStore((s) => s.clearActiveInstance);
  const clearAllInstances = useWorkspaceStore((s) => s.clearAllInstances);
  const upsertSaved = useWorkspaceStore((s) => s.upsertSaved);
  const current = useWorkspaceStore((s) => s.images[s.index] ?? null);
  const work = useWorkspaceStore((s) => {
    const img = s.images[s.index];
    if (!img) return EMPTY_WORK;
    return s.workByImageId[img.id] ?? EMPTY_WORK;
  });

  const active = useMemo(() => {
    return (
      work.instances.find((i) => i.id === work.activeInstanceId) ??
      work.instances[0] ??
      null
    );
  }, [work]);

  const activeIndex = useMemo(() => {
    if (!active) return 0;
    return Math.max(
      0,
      work.instances.findIndex((i) => i.id === active.id),
    );
  }, [work.instances, active]);

  const [busy, setBusy] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!renamingId) return;
    const el = renameInputRef.current;
    if (!el) return;
    el.focus();
    el.select();
  }, [renamingId]);

  function startRename(inst: { id: string; name: string }) {
    setRenamingId(inst.id);
    setRenameDraft(inst.name);
  }

  function commitRename() {
    if (!renamingId) return;
    const trimmed = renameDraft.trim();
    const inst = work.instances.find((i) => i.id === renamingId);
    if (inst) {
      updateInstance(renamingId, {
        name: trimmed || `Instance ${inst.label}`,
      });
    }
    setRenamingId(null);
  }

  function cancelRename() {
    setRenamingId(null);
  }

  useEffect(() => {
    const unsub = useWorkspaceStore.persist.onFinishHydration(() => {
      useWorkspaceStore.getState().setHasHydrated(true);
    });
    if (useWorkspaceStore.persist.hasHydrated()) {
      useWorkspaceStore.getState().setHasHydrated(true);
    }
    return unsub;
  }, []);

  const nmPerPx = current?.nmPerPx ?? "";

  const effectiveNmPerPx = useMemo(() => {
    const n = Number(nmPerPx);
    if (Number.isFinite(n) && n > 0) return n;
    const fromFile = current?.nmPerPxFromFile;
    return fromFile != null && fromFile > 0 ? fromFile : null;
  }, [nmPerPx, current?.nmPerPxFromFile]);

  const effectiveNmPerPxSource = useMemo(():
    | "manual"
    | "zeiss-smartsem"
    | "fei"
    | "imagej"
    | null => {
    if (effectiveNmPerPx == null) return null;
    if (
      current?.nmPerPxFromFile != null &&
      Math.abs(effectiveNmPerPx - current.nmPerPxFromFile) < 1e-9 &&
      current.nmPerPxSource
    ) {
      return current.nmPerPxSource;
    }
    return "manual";
  }, [
    effectiveNmPerPx,
    current?.nmPerPxFromFile,
    current?.nmPerPxSource,
  ]);

  const activeMeasurement = useMemo(() => {
    if (!active?.prediction) return null;
    return measureCrystal(active.prediction, effectiveNmPerPx);
  }, [active, effectiveNmPerPx]);

  const readyCount = useMemo(
    () => instancesReadyToSave(work.instances).length,
    [work.instances],
  );

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
      const withScale = loaded.find(
        (img) => img.nmPerPxFromFile != null && img.nmPerPxFromFile > 0,
      );
      const scaleNote = withScale?.nmPerPxSource
        ? ` · ${resolutionSourceLabel(withScale.nmPerPxSource)}`
        : "";
      toast.success(
        `Loaded ${loaded.length} image${loaded.length === 1 ? "" : "s"}${scaleNote}`,
        { id: loadingId },
      );
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
        const res = await fetch(
          `/api/samples/crystals/${encodeURIComponent(name)}`,
        );
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
      toast.error(
        err instanceof Error ? err.message : "Failed to load samples",
        { id: loadingId },
      );
    }
  }

  async function segmentActive() {
    if (!current || !active || active.points.length === 0) return;
    setBusy(true);
    const loadingId = toast.loading(
      `Segmenting ${active.name}…`,
    );
    try {
      const res = await runVisualSegment({
        image: { type: "base64", value: stripDataUrl(current.dataUrl) },
        points: active.points,
        onStatus: (msg) => toast.loading(msg, { id: loadingId }),
      });
      const pred =
        res.predictions?.[0] ??
        res.prompt_results?.[0]?.predictions?.[0] ??
        null;
      if (!pred) {
        throw new Error("No mask returned from SAM3");
      }
      updateActiveInstance({
        prediction: pred,
        polygons: extractPolygons(pred),
      });
      toast.success(`${active.name} in ${formatNumber(res.time * 1000, 0)} ms`, {
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

  async function segmentAllPending() {
    if (!current) return;
    const pending = work.instances.filter((inst) => inst.points.length > 0);
    if (!pending.length) {
      toast.error("Add click points to at least one instance first.");
      return;
    }
    setBusy(true);
    const loadingId = toast.loading(
      `Segmenting ${pending.length} instance${pending.length === 1 ? "" : "s"}…`,
    );
    try {
      const res = await runVisualSegment({
        image: { type: "base64", value: stripDataUrl(current.dataUrl) },
        promptGroups: pending.map((inst) => inst.points),
        onStatus: (msg) => toast.loading(msg, { id: loadingId }),
      });

      const byPrompt = new Map<number, (typeof res.prompt_results)[number]>();
      for (const pr of res.prompt_results ?? []) {
        byPrompt.set(pr.prompt_index, pr);
      }

      pending.forEach((inst, promptIndex) => {
        const pred =
          byPrompt.get(promptIndex)?.predictions?.[0] ??
          res.predictions?.[promptIndex] ??
          null;
        if (!pred) return;
        updateInstance(inst.id, {
          prediction: pred,
          polygons: extractPolygons(pred),
        });
      });

      toast.success(
        `Segmented ${pending.length} instance${pending.length === 1 ? "" : "s"} in ${formatNumber(res.time * 1000, 0)} ms`,
        { id: loadingId },
      );
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Segmentation failed", {
        id: loadingId,
      });
    } finally {
      setBusy(false);
    }
  }

  /**
   * Few-shot PCS: use ready instance bboxes as visual exemplars, find every
   * matching object on this image (Roboflow concept_segment).
   */
  async function segmentTransfer() {
    if (!current) return;
    const seeds = instancesReadyToSave(work.instances);
    if (!seeds.length) {
      toast.error(
        "Segment at least one instance first, then run Transfer.",
      );
      return;
    }

    const seedBoxes: Array<{
      seed: SegmentInstance;
      box: { x0: number; y0: number; x1: number; y1: number };
      xyxy: number[];
    }> = [];
    for (const seed of seeds) {
      const xyxy = instanceBBoxXyxy(seed);
      if (!xyxy) continue;
      seedBoxes.push({
        seed,
        xyxy,
        box: { x0: xyxy[0], y0: xyxy[1], x1: xyxy[2], y1: xyxy[3] },
      });
    }
    if (!seedBoxes.length) {
      toast.error("Could not get bounding boxes from segmented instances.");
      return;
    }

    setBusy(true);
    const loadingId = toast.loading(
      `Transfer from ${seedBoxes.length} example${seedBoxes.length === 1 ? "" : "s"}…`,
    );
    try {
      const res = await runConceptSegment({
        image: { type: "base64", value: stripDataUrl(current.dataUrl) },
        prompts: [
          {
            type: "visual",
            boxes: seedBoxes.map((s) => s.box),
            box_labels: seedBoxes.map(() => 1),
          },
        ],
        format: "json",
        onStatus: (msg) => toast.loading(msg, { id: loadingId }),
      });

      // Prefer per-prompt predictions; fall back to flat list
      const preds =
        res.prompt_results?.[0]?.predictions ?? res.predictions ?? [];

      const SEED_IOU = 0.5;
      const seedXyxy = seedBoxes.map((s) => s.xyxy);
      const novel = preds.filter((pred) => {
        const bb = predictionBBoxXyxy(pred);
        if (!bb) return false;
        return !seedXyxy.some((sb) => boxIoU(bb, sb) >= SEED_IOU);
      });

      // Keep non-ready instances (in-progress clicks) + seeds + novel matches
      const unfinished = work.instances.filter(
        (inst) =>
          !seeds.some((s) => s.id === inst.id) &&
          (inst.points.length > 0 || inst.polygons.length > 0),
      );

      const added: SegmentInstance[] = novel.map((pred, i) => {
        const base = createEmptyInstance(seeds.length + unfinished.length + i + 1);
        return {
          ...base,
          name: `Instance ${seeds.length + unfinished.length + i + 1}`,
          prediction: pred,
          polygons: extractPolygons(pred),
          points: [],
        };
      });

      const next = [...seeds, ...unfinished, ...added];
      setInstances(next, seeds[0]?.id);
      toast.success(
        `Found ${novel.length} new · ${next.length} total in ${formatNumber(res.time * 1000, 0)} ms`,
        { id: loadingId },
      );
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Concept segmentation failed",
        { id: loadingId },
      );
    } finally {
      setBusy(false);
    }
  }

  async function saveAnnotation() {
    if (!current || readyCount === 0) return;
    setBusy(true);
    try {
      const { metaFileName, maskFileName, metaJson, maskPng, meta } =
        await buildAnnotationDownload({
          imageId: current.id,
          imageName: current.name,
          width: current.width,
          height: current.height,
          instances: work.instances,
          nmPerPx: effectiveNmPerPx,
          nmPerPxSource: effectiveNmPerPxSource,
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
      downloadBlob(
        new Blob([metaJson], { type: "application/json" }),
        metaFileName,
      );
      downloadBlob(maskPng, maskFileName);
      toast.success(
        `Saved ${readyCount} instance${readyCount === 1 ? "" : "s"} → ${metaFileName}`,
      );
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to save annotation",
      );
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

  const activeHasContent =
    !!active && (active.points.length > 0 || active.polygons.length > 0);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden p-4 md:p-6 lg:flex-row">
      <aside className="flex max-h-[42vh] w-full shrink-0 flex-col gap-3 lg:max-h-none lg:h-full lg:w-[296px] lg:min-h-0">
        <header className="flex shrink-0 items-center justify-between gap-2 px-1">
          <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">
            sam4xtal
          </h1>
          <ThemeToggle />
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-1 pb-4 [scrollbar-gutter:stable] lg:pb-2">
          <div className="flex flex-col gap-2">
          <Card
            size="sm"
            className="gap-0 overflow-visible border border-border/60 py-3 shadow-none ring-0"
          >
            <CardContent className="flex flex-col gap-2.5">
              <div className="flex items-center gap-2">
                <p className="shrink-0 text-sm font-medium">Image</p>
                {current ? (
                  <>
                    <p
                      className="min-w-0 flex-1 truncate text-center text-xs text-muted-foreground"
                      title={current.name}
                    >
                      {current.name}
                    </p>
                    <span className="shrink-0 text-xs text-muted-foreground">
                      {index + 1}/{images.length}
                    </span>
                  </>
                ) : null}
              </div>
              <div className="flex gap-1.5">
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
                  className="min-w-0 flex-1"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <FolderOpen />
                  <span className="truncate">Load</span>
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="min-w-0 flex-1"
                  onClick={loadSampleFiles}
                  disabled={busy}
                >
                  <Images />
                  <span className="truncate">Samples</span>
                </Button>
              </div>
              {current ? (
                <div className="flex gap-1.5">
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
              ) : null}
            </CardContent>
          </Card>

          <Card
            size="sm"
            className="gap-0 overflow-visible border border-border/60 py-3 shadow-none ring-0"
          >
            <CardContent className="flex flex-col gap-2">
              <p className="text-sm font-medium">Resolution</p>
              <div className="relative">
                <Input
                  id="nm"
                  type="number"
                  min={0}
                  step="any"
                  placeholder="e.g. 2.5"
                  className="pr-14"
                  value={nmPerPx}
                  disabled={!current}
                  onChange={(e) => setNmPerPx(e.target.value)}
                />
                <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-xs text-muted-foreground">
                  nm/px
                </span>
              </div>
              {effectiveNmPerPxSource && effectiveNmPerPxSource !== "manual" ? (
                <p className="text-[11px] text-muted-foreground">
                  {resolutionSourceLabel(effectiveNmPerPxSource)}
                </p>
              ) : (
                <p className="text-[11px] text-muted-foreground">
                  Per image · from TIFF when available
                </p>
              )}
            </CardContent>
          </Card>

          <Card
            size="sm"
            className="gap-0 overflow-visible border border-border/60 py-3 shadow-none ring-0"
          >
            <CardContent className="flex flex-col gap-2">
              <div className="flex items-center gap-1">
                <p className="mr-auto text-sm font-medium">Instances</p>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  onClick={addInstance}
                  disabled={!current || busy}
                  title="New instance"
                >
                  <Plus />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => active && removeInstance(active.id)}
                  disabled={!current || !active || busy}
                  title="Delete active instance"
                >
                  <Trash2 />
                </Button>
              </div>
              <div className="flex max-h-36 flex-col gap-1.5 overflow-y-auto p-1">
                {work.instances.map((inst, idx) => {
                  const color = colorForInstance(idx);
                  const isRenaming = renamingId === inst.id;
                  return (
                    <div
                      key={inst.id}
                      className="flex items-center gap-2.5"
                      onClick={() => {
                        if (!isRenaming) selectInstance(inst.id);
                      }}
                    >
                      <span
                        className="size-2.5 shrink-0 rounded-full"
                        style={{ backgroundColor: color.solid }}
                        aria-hidden
                      />
                      <Input
                        ref={isRenaming ? renameInputRef : undefined}
                        value={isRenaming ? renameDraft : inst.name}
                        readOnly={!isRenaming}
                        onChange={(e) => setRenameDraft(e.target.value)}
                        onBlur={() => {
                          if (isRenaming) commitRename();
                        }}
                        onKeyDown={(e) => {
                          if (!isRenaming) return;
                          if (e.key === "Enter") {
                            e.preventDefault();
                            commitRename();
                          } else if (e.key === "Escape") {
                            e.preventDefault();
                            cancelRename();
                          }
                        }}
                        onClick={(e) => {
                          e.stopPropagation();
                          selectInstance(inst.id);
                          if (!isRenaming) startRename(inst);
                        }}
                        className="min-w-0 flex-1 focus-visible:ring-inset"
                        aria-label="Instance name"
                        title={isRenaming ? undefined : "Click to rename"}
                      />
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          <Card
            size="sm"
            className="gap-0 overflow-visible border border-border/60 py-3 shadow-none ring-0"
          >
            <CardContent className="flex flex-col gap-2">
              <p className="text-sm font-medium">Prompt</p>
              <div className="grid grid-cols-2 gap-1.5">
                <Button
                  type="button"
                  variant={clickMode === "off" ? "default" : "outline"}
                  size="sm"
                  className="px-1.5"
                  onClick={() => setClickMode("off")}
                >
                  <MousePointer2 />
                  Off
                </Button>
                <DropdownMenu>
                  <DropdownMenuTrigger
                    asChild
                    disabled={
                      busy ||
                      !work.instances.some(
                        (i) => i.points.length > 0 || i.polygons.length > 0,
                      )
                    }
                  >
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="px-1.5"
                    >
                      <Eraser />
                      Clear
                      <ChevronDown className="opacity-70" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-auto min-w-0">
                    <DropdownMenuItem
                      disabled={!activeHasContent || busy}
                      onClick={() => clearActiveInstance()}
                    >
                      Active
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      disabled={
                        busy ||
                        !work.instances.some(
                          (i) => i.points.length > 0 || i.polygons.length > 0,
                        )
                      }
                      onClick={() => clearAllInstances()}
                    >
                      All
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
                <Button
                  type="button"
                  variant={clickMode === "positive" ? "default" : "outline"}
                  size="sm"
                  className="px-1.5"
                  onClick={() => setClickMode("positive")}
                >
                  <CirclePlus />
                  Pos
                </Button>
                <Button
                  type="button"
                  variant={clickMode === "negative" ? "default" : "outline"}
                  size="sm"
                  className="px-1.5"
                  onClick={() => setClickMode("negative")}
                >
                  <CircleMinus />
                  Neg
                </Button>
              </div>
              <div className="mt-1 flex gap-1.5 border-t pt-2">
                <DropdownMenu>
                  <DropdownMenuTrigger
                    asChild
                    disabled={
                      !current ||
                      busy ||
                      (!work.instances.some((i) => i.points.length > 0) &&
                        readyCount === 0)
                    }
                  >
                    <Button
                      type="button"
                      size="sm"
                      className="min-w-0 flex-1"
                    >
                      {busy ? (
                        <Loader2 className="animate-spin" />
                      ) : (
                        <Sparkles />
                      )}
                      <span className="truncate">Segment</span>
                      <ChevronDown className="opacity-70" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-auto min-w-0">
                    <DropdownMenuItem
                      disabled={!active || active.points.length === 0 || busy}
                      onClick={() => void segmentActive()}
                    >
                      Active
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      disabled={
                        busy ||
                        !work.instances.some((i) => i.points.length > 0)
                      }
                      onClick={() => void segmentAllPending()}
                    >
                      All
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      disabled={!current || readyCount === 0 || busy}
                      title="Use segmented instances as exemplars and transfer the concept to the rest of this image"
                      onClick={() => void segmentTransfer()}
                    >
                      Transfer
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  className="min-w-0 flex-1"
                  onClick={saveAnnotation}
                  disabled={readyCount === 0 || busy}
                >
                  <Save />
                  <span className="truncate">
                    Save
                    {readyCount > 0 ? ` (${readyCount})` : ""}
                  </span>
                </Button>
              </div>
            </CardContent>
          </Card>

          {activeMeasurement && (
            <Card
              size="sm"
              className="gap-0 overflow-visible border border-border/60 py-3 shadow-none ring-0"
            >
              <CardContent className="space-y-1 text-sm">
                <p className="flex items-center gap-2 font-medium">
                  <span
                    className="size-2.5 rounded-full"
                    style={{
                      backgroundColor: colorForInstance(activeIndex).solid,
                    }}
                  />
                  {active?.name ?? "Instance"}
                </p>
                <div className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-0.5 font-mono text-xs tabular-nums">
                  <span className="font-sans text-muted-foreground">Area</span>
                  <span>
                    {formatNumber(activeMeasurement.areaPx, 0)} px²
                  </span>
                  <span className="font-sans text-muted-foreground">
                    Eq. diam.
                  </span>
                  <span>
                    {formatNumber(activeMeasurement.equivDiameterPx, 1)} px
                  </span>
                  <span className="font-sans text-muted-foreground">BBox</span>
                  <span>
                    {formatNumber(activeMeasurement.bboxWidthPx, 0)} ×{" "}
                    {formatNumber(activeMeasurement.bboxHeightPx, 0)} px
                  </span>
                  {activeMeasurement.areaNm2 != null && (
                    <>
                      <span className="font-sans text-muted-foreground">
                        Area
                      </span>
                      <span>
                        {formatNumber(activeMeasurement.areaNm2, 0)} nm²
                      </span>
                      <span className="font-sans text-muted-foreground">
                        Eq. diam.
                      </span>
                      <span>
                        {formatNumber(
                          activeMeasurement.equivDiameterNm ?? 0,
                          1,
                        )}{" "}
                        nm
                      </span>
                      <span className="font-sans text-muted-foreground">
                        BBox
                      </span>
                      <span>
                        {formatNumber(activeMeasurement.bboxWidthNm ?? 0, 1)} ×{" "}
                        {formatNumber(activeMeasurement.bboxHeightNm ?? 0, 1)} nm
                      </span>
                    </>
                  )}
                  <span className="font-sans text-muted-foreground">
                    Confidence
                  </span>
                  <span>
                    {formatNumber(activeMeasurement.confidence * 100, 1)}%
                  </span>
                </div>
              </CardContent>
            </Card>
          )}
          </div>
        </div>
      </aside>

        <Card className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden border border-border/60 py-0 shadow-none ring-0">
          <CardContent className="min-h-0 flex-1 p-0">
            <div className="h-full min-h-0">
              {current ? (
                <ImageCanvas
                  src={current.dataUrl}
                  instances={work.instances}
                  activeInstanceId={work.activeInstanceId}
                  clickMode={clickMode}
                  disabled={busy}
                  onAddPoint={addPoint}
                  onRemovePoint={removePoint}
                  onSelectInstance={selectInstance}
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
  );
}
