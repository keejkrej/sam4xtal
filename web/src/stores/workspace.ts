"use client";

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { cacheImages, hydrateImages, type ImageRef } from "@/lib/image-cache";
import type {
  AnnotationResult,
  ImageWork,
  PointPrompt,
  SegmentationPrediction,
  WorkspaceImage,
} from "@/lib/types";

const EMPTY_WORK: ImageWork = {
  points: [],
  polygons: [],
  prediction: null,
};

const WORKSPACE_KEY = "sam4xtal-workspace";

/** Keep session payload small — drop mask tensors; polygons/points are enough. */
function lightWork(work: ImageWork): ImageWork {
  if (!work.prediction) return work;
  return {
    ...work,
    prediction: { ...work.prediction, masks: [] },
  };
}

type WorkspaceState = {
  images: WorkspaceImage[];
  index: number;
  nmPerPx: string;
  negativeMode: boolean;
  workByImageId: Record<string, ImageWork>;
  saved: AnnotationResult[];
  hasHydrated: boolean;
  setHasHydrated: (v: boolean) => void;
  setImages: (images: WorkspaceImage[]) => void;
  setIndex: (index: number) => void;
  goTo: (index: number) => void;
  setNmPerPx: (v: string) => void;
  setNegativeMode: (v: boolean) => void;
  addPoint: (point: PointPrompt) => void;
  setWork: (work: Partial<ImageWork>) => void;
  clearCurrentWork: () => void;
  upsertSaved: (annotation: AnnotationResult) => void;
  currentImage: () => WorkspaceImage | null;
  currentWork: () => ImageWork;
};

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set, get) => ({
      images: [],
      index: 0,
      nmPerPx: "",
      negativeMode: false,
      workByImageId: {},
      saved: [],
      hasHydrated: false,
      setHasHydrated: (v) => set({ hasHydrated: v }),
      setImages: (images) => {
        void cacheImages(images);
        set({
          images,
          index: 0,
          workByImageId: {},
        });
      },
      setIndex: (index) => set({ index }),
      goTo: (index) => {
        const { images } = get();
        if (!images.length) return;
        const clamped = Math.max(0, Math.min(images.length - 1, index));
        set({ index: clamped });
      },
      setNmPerPx: (nmPerPx) => set({ nmPerPx }),
      setNegativeMode: (negativeMode) => set({ negativeMode }),
      addPoint: (point) => {
        const img = get().currentImage();
        if (!img) return;
        const prev = get().workByImageId[img.id] ?? EMPTY_WORK;
        set({
          workByImageId: {
            ...get().workByImageId,
            [img.id]: { ...prev, points: [...prev.points, point] },
          },
        });
      },
      setWork: (work) => {
        const img = get().currentImage();
        if (!img) return;
        const prev = get().workByImageId[img.id] ?? EMPTY_WORK;
        set({
          workByImageId: {
            ...get().workByImageId,
            [img.id]: { ...prev, ...work },
          },
        });
      },
      clearCurrentWork: () => {
        const img = get().currentImage();
        if (!img) return;
        const { [img.id]: _removed, ...rest } = get().workByImageId;
        set({ workByImageId: rest });
      },
      upsertSaved: (annotation) =>
        set((state) => ({
          saved: [
            ...state.saved.filter((r) => r.imageId !== annotation.imageId),
            annotation,
          ],
        })),
      currentImage: () => {
        const { images, index } = get();
        return images[index] ?? null;
      },
      currentWork: () => {
        const img = get().currentImage();
        if (!img) return EMPTY_WORK;
        return get().workByImageId[img.id] ?? EMPTY_WORK;
      },
    }),
    {
      name: WORKSPACE_KEY,
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        images: state.images.map(
          ({ dataUrl: _dataUrl, ...ref }): ImageRef => ref,
        ),
        index: state.index,
        nmPerPx: state.nmPerPx,
        negativeMode: state.negativeMode,
        workByImageId: Object.fromEntries(
          Object.entries(state.workByImageId).map(([id, work]) => [
            id,
            lightWork(work),
          ]),
        ),
        saved: state.saved,
      }),
      onRehydrateStorage: () => (state) => {
        void (async () => {
          try {
            // Let the store finish initializing before touching useWorkspaceStore.
            await Promise.resolve();
            if (state?.images?.length) {
              const refs = state.images as unknown as ImageRef[];
              const images = await hydrateImages(refs);
              const ids = new Set(images.map((img) => img.id));
              useWorkspaceStore.setState({
                images,
                index: images.length
                  ? Math.min(state.index, images.length - 1)
                  : 0,
                workByImageId: Object.fromEntries(
                  Object.entries(state.workByImageId).filter(([id]) =>
                    ids.has(id),
                  ),
                ),
              });
            }
          } finally {
            useWorkspaceStore.getState().setHasHydrated(true);
          }
        })();
      },
    },
  ),
);

export { EMPTY_WORK };
export type { PointPrompt, SegmentationPrediction };
