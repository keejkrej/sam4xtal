import { readdir } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const IMAGE_RE = /\.(png|jpe?g|tif?f|bmp|webp)$/i;

function crystalsDir(): string {
  return path.resolve(process.cwd(), "..", "samples", "crystals");
}

export async function GET() {
  try {
    const entries = await readdir(crystalsDir());
    const files = entries
      .filter((name) => IMAGE_RE.test(name))
      .sort((a, b) => a.localeCompare(b));
    return NextResponse.json({ files });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Failed to list samples";
    return NextResponse.json({ error: message, files: [] }, { status: 500 });
  }
}
