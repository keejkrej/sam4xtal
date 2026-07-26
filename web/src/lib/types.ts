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
};

export type ImageWork = {
  points: PointPrompt[];
  polygons: number[][][];
  prediction: SegmentationPrediction | null;
};

export type CrystalMeasurement = {
  areaPx: number;
  equivDiameterPx: number;
  areaNm2: number | null;
  equivDiameterNm: number | null;
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

export type AnnotationResult = {
  imageId: string;
  imageName: string;
  imageWidth: number;
  imageHeight: number;
  maskFileName: string;
  points: PointPrompt[];
  measurement: CrystalMeasurement;
  nmPerPx: number | null;
  bbox_xyxy?: number[];
  confidence: number;
  savedAt: string;
};
