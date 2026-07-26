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

export type SemImage = {
  id: string;
  name: string;
  file: File;
  objectUrl: string;
  width: number;
  height: number;
};

export type CrystalMeasurement = {
  areaPx: number;
  equivDiameterPx: number;
  areaNm2: number | null;
  equivDiameterNm: number | null;
  confidence: number;
};

export type AnnotationResult = {
  imageId: string;
  imageName: string;
  points: PointPrompt[];
  maskPolygons: number[][][];
  measurement: CrystalMeasurement;
  nmPerPx: number | null;
  savedAt: string;
};
