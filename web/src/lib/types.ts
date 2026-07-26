/** Types aligned with Roboflow Inference SAM3 HTTP API. */

export type InferenceImage =
  | { type: "base64"; value: string }
  | { type: "url"; value: string };

export type PointPrompt = {
  x: number;
  y: number;
  positive: boolean;
};

export type VisualPrompt = {
  points?: PointPrompt[];
  box?: { x: number; y: number; width: number; height: number };
};

export type SegmentationPrediction = {
  format: string;
  confidence: number;
  masks: number[][][] | unknown[];
  area_px?: number;
  bbox_xyxy?: number[];
};

export type VisualSegmentResponse = {
  time: number;
  predictions: SegmentationPrediction[];
  prompt_results: Array<{
    prompt_index: number;
    predictions: SegmentationPrediction[];
  }>;
};

export type WorkspaceImage = {
  id: string;
  name: string;
  width: number;
  height: number;
  mimeType: string;
  /** data: URL — held in memory; durable copy lives in the local image cache */
  dataUrl: string;
  /**
   * Pixel scale read from file metadata when available (e.g. Zeiss SmartSEM
   * TIFF tag 34118 "Image Pixel Size"). Null if absent / not a tagged TIFF.
   */
  nmPerPxFromFile?: number | null;
  /** Provenance for `nmPerPxFromFile`, when set. */
  nmPerPxSource?: "zeiss-smartsem" | "fei" | "imagej" | null;
};

/** One crystal / object instance on an image. */
export type SegmentInstance = {
  id: string;
  /** 1-based display / metadata label. */
  label: number;
  /** User-editable display name (defaults to `Instance ${label}`). */
  name: string;
  points: PointPrompt[];
  polygons: number[][][];
  prediction: SegmentationPrediction | null;
};

export type ImageWork = {
  instances: SegmentInstance[];
  activeInstanceId: string | null;
};

export type CrystalMeasurement = {
  areaPx: number;
  equivDiameterPx: number;
  bboxWidthPx: number;
  bboxHeightPx: number;
  areaNm2: number | null;
  equivDiameterNm: number | null;
  bboxWidthNm: number | null;
  bboxHeightNm: number | null;
  confidence: number;
};

export type SidecarHealth = {
  ok: boolean;
  ready: boolean;
  load_state: string;
  backend?: string;
  model_loaded?: boolean;
  model_id?: string;
  device?: string;
  error?: string | null;
};

/** RGB identity color written into the exported mask PNG for one instance. */
export type InstanceMaskColor = {
  r: number;
  g: number;
  b: number;
  /** Lowercase `#rrggbb` matching the mask pixels. */
  hex: string;
};

export type InstanceAnnotation = {
  id: string;
  label: number;
  name: string;
  /** Color used for this instance in `maskFileName` (bg is `#000000`). */
  color: InstanceMaskColor;
  points: PointPrompt[];
  measurement: CrystalMeasurement;
  bbox_xyxy?: number[];
  confidence: number;
};

export type AnnotationResult = {
  imageId: string;
  imageName: string;
  imageWidth: number;
  imageHeight: number;
  maskFileName: string;
  /**
   * RGB colormap mask: background `#000000`; each instance paints a unique
   * colormap color recorded on `instances[].color` for downstream filtering.
   */
  maskEncoding: "instance-colors";
  backgroundColor: InstanceMaskColor;
  instances: InstanceAnnotation[];
  /** SEM scale used for nm measurements (manual or from file metadata). */
  nmPerPx: number | null;
  /**
   * Where `nmPerPx` came from when saved. `manual` if the user typed it;
   * instrument tags when read from the image; omitted/null if unset.
   */
  nmPerPxSource?: "manual" | "zeiss-smartsem" | "fei" | "imagej" | null;
  savedAt: string;
};
