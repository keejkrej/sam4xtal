import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const IMAGE_RE = /\.(png|jpe?g|tif?f|bmp|webp)$/i;

const MIME: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".tif": "image/tiff",
  ".tiff": "image/tiff",
  ".bmp": "image/bmp",
  ".webp": "image/webp",
};

function crystalsDir(): string {
  return path.resolve(process.cwd(), "..", "samples", "crystals");
}

export async function GET(
  _req: Request,
  ctx: { params: Promise<{ name: string }> },
) {
  const { name: raw } = await ctx.params;
  const name = path.basename(decodeURIComponent(raw));
  if (name !== raw.replace(/\\/g, "/") || !IMAGE_RE.test(name)) {
    return new NextResponse("Not found", { status: 404 });
  }

  try {
    const filePath = path.join(crystalsDir(), name);
    const data = await readFile(filePath);
    const ext = path.extname(name).toLowerCase();
    return new NextResponse(data, {
      headers: {
        "Content-Type": MIME[ext] ?? "application/octet-stream",
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch {
    return new NextResponse("Not found", { status: 404 });
  }
}
