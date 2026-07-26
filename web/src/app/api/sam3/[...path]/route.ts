import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const DEFAULT_INFERENCE_URL = "http://127.0.0.1:9001";

function inferenceBaseUrl(): string {
  return (
    process.env.INFERENCE_URL?.replace(/\/$/, "") ||
    process.env.SAM3_SIDECAR_URL?.replace(/\/$/, "") ||
    DEFAULT_INFERENCE_URL
  );
}

function unreachablePayload(target: string, message: string) {
  return {
    detail: {
      message:
        "SAM3 sidecar is unreachable — it may still be starting or downloading model weights.",
      load_state: "unreachable",
      target,
      cause: message,
    },
    hint: "Wait for the sidecar to finish loading, or start it with `.\\sidecar\\run.ps1`.",
  };
}

/**
 * Proxy to the local SAM3 sidecar (or Roboflow serverless).
 *
 * Local:   INFERENCE_URL=http://127.0.0.1:9001
 * Roboflow: INFERENCE_URL=https://serverless.roboflow.com
 *           ROBOFLOW_API_KEY=...
 *
 * Paths match https://inference.roboflow.com/foundation/sam3/
 * e.g. /api/sam3/visual_segment → {INFERENCE_URL}/sam3/visual_segment
 * Special: /api/sam3/health → {INFERENCE_URL}/health
 */
export async function POST(
  req: NextRequest,
  ctx: { params: Promise<{ path: string[] }> },
) {
  const { path } = await ctx.params;
  const segments = path ?? [];
  const targetPath = `/sam3/${segments.join("/")}`;
  const base = inferenceBaseUrl();

  const url = new URL(`${base}${targetPath}`);
  const apiKey = process.env.ROBOFLOW_API_KEY;
  if (apiKey) {
    url.searchParams.set("api_key", apiKey);
  }

  const incomingKey = req.nextUrl.searchParams.get("api_key");
  if (incomingKey) {
    url.searchParams.set("api_key", incomingKey);
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const upstream = await fetch(url.toString(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const text = await upstream.text();
    const contentType = upstream.headers.get("content-type") || "application/json";
    return new NextResponse(text, {
      status: upstream.status,
      headers: { "Content-Type": contentType },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Upstream request failed";
    return NextResponse.json(unreachablePayload(`${base}${targetPath}`, message), {
      status: 503,
    });
  }
}

export async function GET(
  _req: NextRequest,
  ctx: { params: Promise<{ path: string[] }> },
) {
  const { path } = await ctx.params;
  const segments = path ?? [];
  const base = inferenceBaseUrl();

  // /api/sam3/health → sidecar /health (readiness while weights download)
  if (segments.length === 1 && segments[0] === "health") {
    try {
      const upstream = await fetch(`${base}/health`, { cache: "no-store" });
      const text = await upstream.text();
      return new NextResponse(text, {
        status: upstream.status,
        headers: {
          "Content-Type": upstream.headers.get("content-type") || "application/json",
        },
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upstream request failed";
      return NextResponse.json(unreachablePayload(`${base}/health`, message), {
        status: 503,
      });
    }
  }

  return NextResponse.json({
    ok: true,
    inferenceUrl: base,
    endpoints: [
      "/api/sam3/health",
      "/api/sam3/embed_image",
      "/api/sam3/visual_segment",
      "/api/sam3/concept_segment",
    ],
  });
}
