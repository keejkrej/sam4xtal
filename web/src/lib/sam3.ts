import type {
  CrystalMeasurement,
  InferenceImage,
  PointPrompt,
  SegmentationPrediction,
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

export async function runVisualSegment(args: {
  image: InferenceImage;
  points: PointPrompt[];
  multimaskOutput?: boolean;
}): Promise<VisualSegmentResponse> {
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
    throw new Error(detail || `Segmentation failed (${res.status})`);
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
