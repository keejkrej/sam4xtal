"use client";

import { del, get, set as idbSet } from "idb-keyval";
import type { WorkspaceImage } from "@/lib/types";

const IMAGE_CACHE_PREFIX = "sam4xtal-image:";

export type ImageRef = Omit<WorkspaceImage, "dataUrl">;

function cacheKey(id: string): string {
  return `${IMAGE_CACHE_PREFIX}${id}`;
}

/** Durable local cache for image bytes (IndexedDB). Shared across tabs. */
export async function cacheImage(id: string, dataUrl: string): Promise<void> {
  await idbSet(cacheKey(id), dataUrl);
}

export async function cacheImages(images: WorkspaceImage[]): Promise<void> {
  await Promise.all(images.map((img) => cacheImage(img.id, img.dataUrl)));
}

export async function getCachedImage(id: string): Promise<string | null> {
  return (await get<string>(cacheKey(id))) ?? null;
}

export async function hydrateImages(refs: ImageRef[]): Promise<WorkspaceImage[]> {
  const hydrated: WorkspaceImage[] = [];
  for (const ref of refs) {
    const dataUrl = await getCachedImage(ref.id);
    if (dataUrl) {
      hydrated.push({ ...ref, dataUrl });
    }
  }
  return hydrated;
}

export async function dropCachedImage(id: string): Promise<void> {
  await del(cacheKey(id));
}
