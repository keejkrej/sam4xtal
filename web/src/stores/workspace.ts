"use client";

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { cacheImages, hydrateImages, type ImageRef } from "@/lib/image-cache";
import type {
  AnnotationResult,
  ImageWork,
  PointPrompt,
  SegmentInstance,
  SegmentationPrediction,
  WorkspaceImage,
} from "@/lib/types";

function newInstanceId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `inst-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function createEmptyInstance(label = 1): SegmentInstance {
  return {
    id: newInstanceId(),
    label,
    name: `Instance ${label}`,
    points: [],
    polygons: [],
    prediction: null,
  };
}

function relabel(instances: SegmentInstance[]): SegmentInstance[] {
  return instances.map((inst, i) => {
    const label = i + 1;
    // Keep custom names; refresh default "Instance N" when it still matched the old label.
    const name =
      inst.name && inst.name !== `Instance ${inst.label}`
        ? inst.name
        : `Instance ${label}`;
    return { ...inst, label, name };
  });
}

export function createEmptyWork(): ImageWork {
  const inst = createEmptyInstance(1);
  return { instances: [inst], activeInstanceId: inst.id };
}

const EMPTY_WORK = createEmptyWork();

const WORKSPACE_KEY = "sam4xtal-workspace";

/** Keep session payload small — drop mask tensors; polygons/points are enough. */
function lightInstance(inst: SegmentInstance): SegmentInstance {
  if (!inst.prediction) return inst;
  return {
    ...inst,
    prediction: { ...inst.prediction, masks: [] },
  };
}

function lightWork(work: ImageWork): ImageWork {
  return {
    ...work,
    instances: work.instances.map(lightInstance),
  };
}

/** Migrate pre-multi-instance persisted work `{ points, polygons, prediction }`. */
function normalizeWork(raw: unknown): ImageWork {
  if (!raw || typeof raw !== "object") return createEmptyWork();
  const obj = raw as Record<string, unknown>;

  if (Array.isArray(obj.instances)) {
    const instances = (obj.instances as SegmentInstance[]).map((inst, i) => ({
      id: inst.id || newInstanceId(),
      label: inst.label ?? i + 1,
      name:
        typeof inst.name === "string" && inst.name.trim()
          ? inst.name.trim()
          : `Instance ${inst.label ?? i + 1}`,
      points: Array.isArray(inst.points) ? inst.points : [],
      polygons: Array.isArray(inst.polygons) ? inst.polygons : [],
      prediction: inst.prediction ?? null,
    }));
    if (!instances.length) return createEmptyWork();
    const active =
      typeof obj.activeInstanceId === "string" &&
      instances.some((i) => i.id === obj.activeInstanceId)
        ? (obj.activeInstanceId as string)
        : instances[0].id;
    return { instances: relabel(instances), activeInstanceId: active };
  }

  // Legacy single-mask shape
  const points = Array.isArray(obj.points) ? (obj.points as PointPrompt[]) : [];
  const polygons = Array.isArray(obj.polygons)
    ? (obj.polygons as number[][][])
    : [];
  const prediction =
    (obj.prediction as SegmentationPrediction | null | undefined) ?? null;
  const inst = createEmptyInstance(1);
  inst.points = points;
  inst.polygons = polygons;
  inst.prediction = prediction;
  return { instances: [inst], activeInstanceId: inst.id };
}

type WorkspaceState = {
  images: WorkspaceImage[];
  index: number;
  nmPerPx: string;
  /** off = select instances; positive/negative = place or remove prompt points */
  clickMode: "off" | "positive" | "negative";
  workByImageId: Record<string, ImageWork>;
  saved: AnnotationResult[];
  hasHydrated: boolean;
  setHasHydrated: (v: boolean) => void;
  setImages: (images: WorkspaceImage[]) => void;
  setIndex: (index: number) => void;
  goTo: (index: number) => void;
  setNmPerPx: (v: string) => void;
  setClickMode: (v: "off" | "positive" | "negative") => void;
  addPoint: (point: PointPrompt) => void;
  removePoint: (index: number) => void;
  ensureActiveInstance: () => SegmentInstance;
  addInstance: () => void;
  selectInstance: (id: string) => void;
  removeInstance: (id: string) => void;
  updateActiveInstance: (patch: Partial<SegmentInstance>) => void;
  updateInstance: (id: string, patch: Partial<SegmentInstance>) => void;
  clearActiveInstance: () => void;
  clearAllInstances: () => void;
  upsertSaved: (annotation: AnnotationResult) => void;
  currentImage: () => WorkspaceImage | null;
  currentWork: () => ImageWork;
  activeInstance: () => SegmentInstance | null;
};

function mutateCurrentWork(
  get: () => WorkspaceState,
  set: (partial: Partial<WorkspaceState>) => void,
  updater: (work: ImageWork) => ImageWork,
) {
  const img = get().currentImage();
  if (!img) return;
  const prev = normalizeWork(get().workByImageId[img.id] ?? createEmptyWork());
  set({
    workByImageId: {
      ...get().workByImageId,
      [img.id]: updater(prev),
    },
  });
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set, get) => ({
      images: [],
      index: 0,
      nmPerPx: "",
      clickMode: "off",
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
      setClickMode: (clickMode) => set({ clickMode }),
      ensureActiveInstance: () => {
        const img = get().currentImage();
        if (!img) {
          return createEmptyInstance(1);
        }
        const work = normalizeWork(
          get().workByImageId[img.id] ?? createEmptyWork(),
        );
        let active = work.instances.find((i) => i.id === work.activeInstanceId);
        if (!active) {
          active = work.instances[0] ?? createEmptyInstance(1);
          const instances = work.instances.length ? work.instances : [active];
          set({
            workByImageId: {
              ...get().workByImageId,
              [img.id]: {
                instances: relabel(instances),
                activeInstanceId: active.id,
              },
            },
          });
        }
        return active;
      },
      addPoint: (point) => {
        const img = get().currentImage();
        if (!img) return;
        mutateCurrentWork(get, set, (work) => {
          let instances = work.instances;
          let activeId = work.activeInstanceId;
          let active = instances.find((i) => i.id === activeId);
          if (!active) {
            const created = createEmptyInstance(instances.length + 1);
            instances = [...instances, created];
            activeId = created.id;
            active = created;
          }
          return {
            instances: instances.map((inst) =>
              inst.id === active!.id
                ? { ...inst, points: [...inst.points, point] }
                : inst,
            ),
            activeInstanceId: activeId,
          };
        });
      },
      removePoint: (index) => {
        mutateCurrentWork(get, set, (work) => {
          const activeId = work.activeInstanceId;
          if (!activeId || index < 0) return work;
          return {
            ...work,
            instances: work.instances.map((inst) =>
              inst.id === activeId
                ? {
                    ...inst,
                    points: inst.points.filter((_, i) => i !== index),
                  }
                : inst,
            ),
          };
        });
      },
      addInstance: () => {
        mutateCurrentWork(get, set, (work) => {
          const created = createEmptyInstance(work.instances.length + 1);
          return {
            instances: relabel([...work.instances, created]),
            activeInstanceId: created.id,
          };
        });
      },
      selectInstance: (id) => {
        mutateCurrentWork(get, set, (work) => {
          if (!work.instances.some((i) => i.id === id)) return work;
          return { ...work, activeInstanceId: id };
        });
      },
      removeInstance: (id) => {
        mutateCurrentWork(get, set, (work) => {
          const remaining = work.instances.filter((i) => i.id !== id);
          if (!remaining.length) {
            const created = createEmptyInstance(1);
            return { instances: [created], activeInstanceId: created.id };
          }
          const activeId =
            work.activeInstanceId === id
              ? remaining[remaining.length - 1].id
              : work.activeInstanceId;
          return {
            instances: relabel(remaining),
            activeInstanceId: activeId,
          };
        });
      },
      updateActiveInstance: (patch) => {
        const activeId = get().currentWork().activeInstanceId;
        if (!activeId) return;
        get().updateInstance(activeId, patch);
      },
      updateInstance: (id, patch) => {
        mutateCurrentWork(get, set, (work) => {
          if (!work.instances.some((i) => i.id === id)) return work;
          return {
            ...work,
            instances: work.instances.map((inst) =>
              inst.id === id ? { ...inst, ...patch, id: inst.id } : inst,
            ),
          };
        });
      },
      clearActiveInstance: () => {
        mutateCurrentWork(get, set, (work) => {
          const activeId = work.activeInstanceId;
          if (!activeId) return work;
          return {
            ...work,
            instances: work.instances.map((inst) =>
              inst.id === activeId
                ? { ...inst, points: [], polygons: [], prediction: null }
                : inst,
            ),
          };
        });
      },
      clearAllInstances: () => {
        mutateCurrentWork(get, set, () => createEmptyWork());
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
        if (!img) return createEmptyWork();
        return normalizeWork(get().workByImageId[img.id] ?? createEmptyWork());
      },
      activeInstance: () => {
        const work = get().currentWork();
        return (
          work.instances.find((i) => i.id === work.activeInstanceId) ??
          work.instances[0] ??
          null
        );
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
        clickMode: state.clickMode,
        workByImageId: Object.fromEntries(
          Object.entries(state.workByImageId).map(([id, work]) => [
            id,
            lightWork(normalizeWork(work)),
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
                  Object.entries(state.workByImageId)
                    .filter(([id]) => ids.has(id))
                    .map(([id, work]) => [id, normalizeWork(work)]),
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

export { EMPTY_WORK, normalizeWork };
export type { PointPrompt, SegmentationPrediction };
