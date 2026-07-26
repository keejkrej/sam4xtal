import type {
  AnnotationResult,
  CrystalMeasurement,
  InferenceImage,
  PointPrompt,
  SegmentationPrediction,
  SidecarHealth,
  VisualSegmentResponse,
} from "./types";

function stripDataUrl(dataUrl: string): string {
  const idx = dataUrl.indexOf(",");
  return idx >= 0 ? dataUrl.slice(idx + 1) : dataUrl;
}

export async function fileToBase64Image(file: File): Promise<InferenceImage> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  return { type: "base64", value: stripDataUrl(dataUrl) };
}

export class SidecarBusyError extends Error {
  readonly loadState: string;

  constructor(message: string, loadState = "loading") {
    super(message);
    this.name = "SidecarBusyError";
    this.loadState = loadState;
  }
}

function parseErrorPayload(text: string, status: number): Error {
  try {
    const json = JSON.parse(text) as {
      detail?:
        | string
        | {
            message?: string;
            load_state?: string;
            cause?: string;
          };
      hint?: string;
    };
    const detail = json.detail;
    if (detail && typeof detail === "object") {
      const state = detail.load_state ?? (status === 503 ? "loading" : "error");
      const msg =
        detail.message ||
        (state === "unreachable"
          ? "SAM3 sidecar is starting or downloading model weights."
          : "SAM3 sidecar is not ready yet.");
      if (state === "loading" || state === "unreachable" || status === 503) {
        return new SidecarBusyError(msg, state);
      }
      return new Error(detail.cause ? `${msg} (${detail.cause})` : msg);
    }
    if (typeof detail === "string") {
      if (status === 503 || /download|loading|unreachable|fetch failed/i.test(detail)) {
        return new SidecarBusyError(detail, "loading");
      }
      return new Error(detail);
    }
  } catch {
    // fall through
  }
  if (status === 503) {
    return new SidecarBusyError(
      "SAM3 sidecar is starting or downloading model weights.",
      "loading",
    );
  }
  return new Error(text || `Segmentation failed (${status})`);
}

export async function fetchSidecarHealth(): Promise<SidecarHealth> {
  try {
    const res = await fetch("/api/sam3/health", { cache: "no-store" });
    const json = (await res.json()) as SidecarHealth & {
      detail?: { message?: string; load_state?: string };
    };
    if (!res.ok) {
      return {
        ok: false,
        ready: false,
        load_state: json.detail?.load_state ?? "unreachable",
        backend: json.backend,
        model_id: json.model_id,
        device: json.device,
        error: json.detail?.message ?? json.error ?? `HTTP ${res.status}`,
      };
    }
    return {
      ok: Boolean(json.ok),
      ready: Boolean(json.ready ?? json.model_loaded),
      load_state: json.load_state ?? (json.model_loaded ? "ready" : "loading"),
      backend: json.backend,
      model_loaded: json.model_loaded,
      model_id: json.model_id,
      device: json.device,
      error: json.error,
    };
  } catch (err) {
    return {
      ok: false,
      ready: false,
      load_state: "unreachable",
      error: err instanceof Error ? err.message : "Sidecar unreachable",
    };
  }
}

/** Poll until the sidecar reports ready, or throw after timeout. */
export async function waitForSidecarReady(options?: {
  timeoutMs?: number;
  intervalMs?: number;
  onStatus?: (health: SidecarHealth) => void;
}): Promise<SidecarHealth> {
  const timeoutMs = options?.timeoutMs ?? 10 * 60 * 1000;
  const intervalMs = options?.intervalMs ?? 1500;
  const start = Date.now();

  while (Date.now() - start < timeoutMs) {
    const health = await fetchSidecarHealth();
    options?.onStatus?.(health);
    if (health.ready && health.load_state === "ready") return health;
    if (health.load_state === "error") {
      throw new Error(health.error || "SAM3 model failed to load");
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new SidecarBusyError(
    "Timed out waiting for SAM3 model to finish downloading.",
    "loading",
  );
}

export async function runVisualSegment(args: {
  image: InferenceImage;
  points: PointPrompt[];
  multimaskOutput?: boolean;
  waitForModel?: boolean;
  onStatus?: (message: string) => void;
}): Promise<VisualSegmentResponse> {
  if (args.waitForModel !== false) {
    await waitForSidecarReady({
      onStatus: (health) => {
        if (health.load_state === "loading") {
          args.onStatus?.(
            `Downloading / loading ${health.model_id ?? "SAM3"}…`,
          );
        } else if (health.load_state === "unreachable") {
          args.onStatus?.("Waiting for SAM3 sidecar to start…");
        }
      },
    });
  }

  const res = await fetch("/api/sam3/visual_segment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image: args.image,
      prompts: [
        {
          points: args.points,
        },
      ],
      multimask_output: args.multimaskOutput ?? false,
      format: "json",
    }),
  });

  if (!res.ok) {
    const detail = await res.text();
    const err = parseErrorPayload(detail, res.status);
    if (err instanceof SidecarBusyError && args.waitForModel !== false) {
      args.onStatus?.(err.message);
      await waitForSidecarReady({
        onStatus: (health) => {
          if (health.load_state === "loading") {
            args.onStatus?.(
              `Downloading / loading ${health.model_id ?? "SAM3"}…`,
            );
          }
        },
      });
      return runVisualSegment({ ...args, waitForModel: false });
    }
    throw err;
  }
  return res.json();
}

export function polygonArea(points: number[][]): number {
  if (points.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}

export function extractPolygons(pred: SegmentationPrediction): number[][][] {
  if (!pred?.masks || !Array.isArray(pred.masks)) return [];
  const out: number[][][] = [];
  for (const poly of pred.masks) {
    if (
      Array.isArray(poly) &&
      poly.length > 0 &&
      Array.isArray(poly[0]) &&
      typeof (poly as number[][])[0][0] === "number"
    ) {
      out.push(poly as number[][]);
    }
  }
  return out;
}

export function measureCrystal(
  pred: SegmentationPrediction,
  nmPerPx: number | null,
): CrystalMeasurement {
  const polygons = extractPolygons(pred);
  const areaFromPolys = polygons.reduce((acc, poly) => acc + polygonArea(poly), 0);
  const areaPx = pred.area_px ?? Math.round(areaFromPolys);
  const equivDiameterPx = areaPx > 0 ? 2 * Math.sqrt(areaPx / Math.PI) : 0;

  return {
    areaPx,
    equivDiameterPx,
    areaNm2: nmPerPx != null ? areaPx * nmPerPx * nmPerPx : null,
    equivDiameterNm: nmPerPx != null ? equivDiameterPx * nmPerPx : null,
    confidence: pred.confidence ?? 0,
  };
}

/** Rasterize mask polygons to a single-channel PNG: background 0, foreground 255. */
export async function polygonsToMaskPng(
  polygons: number[][][],
  width: number,
  height: number,
): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not create canvas for mask export");

  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  for (const poly of polygons) {
    if (poly.length < 3) continue;
    ctx.beginPath();
    ctx.moveTo(poly[0][0], poly[0][1]);
    for (let i = 1; i < poly.length; i++) {
      ctx.lineTo(poly[i][0], poly[i][1]);
    }
    ctx.closePath();
    ctx.fill();
  }

  // Force true grayscale 0/255 via ImageData (canvas fill can be slightly anti-aliased).
  const image = ctx.getImageData(0, 0, width, height);
  const { data } = image;
  for (let i = 0; i < data.length; i += 4) {
    const on = data[i] > 127 || data[i + 1] > 127 || data[i + 2] > 127;
    const v = on ? 255 : 0;
    data[i] = v;
    data[i + 1] = v;
    data[i + 2] = v;
    data[i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);

  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, "image/png"),
  );
  if (!blob) throw new Error("Failed to encode mask PNG");
  return blob;
}

/** Build annotation metadata JSON + binary mask PNG (0/255). */
export async function buildAnnotationDownload(args: {
  imageId: string;
  imageName: string;
  width: number;
  height: number;
  points: PointPrompt[];
  polygons: number[][][];
  measurement: CrystalMeasurement;
  nmPerPx: number | null;
  bbox_xyxy?: number[];
}): Promise<{
  metaFileName: string;
  maskFileName: string;
  metaJson: string;
  maskPng: Blob;
  meta: AnnotationResult;
}> {
  const stem = args.imageName.replace(/\.[^.]+$/, "");
  const maskFileName = `${stem}.mask.png`;
  const metaFileName = `${stem}.mask.json`;
  const maskPng = await polygonsToMaskPng(args.polygons, args.width, args.height);
  const meta: AnnotationResult = {
    imageId: args.imageId,
    imageName: args.imageName,
    imageWidth: args.width,
    imageHeight: args.height,
    maskFileName,
    points: args.points,
    measurement: args.measurement,
    nmPerPx: args.nmPerPx,
    bbox_xyxy: args.bbox_xyxy,
    confidence: args.measurement.confidence,
    savedAt: new Date().toISOString(),
  };
  return {
    metaFileName,
    maskFileName,
    metaJson: JSON.stringify(meta, null, 2),
    maskPng,
    meta,
  };
}
