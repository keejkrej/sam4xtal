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

/**
 * Proxy to the local SAM3 sidecar (or Roboflow serverless).
 *
 * Local:   INFERENCE_URL=http://127.0.0.1:9001
 * Roboflow: INFERENCE_URL=https://serverless.roboflow.com
 *           ROBOFLOW_API_KEY=...
 *
 * Paths match https://inference.roboflow.com/foundation/sam3/
 * e.g. /api/sam3/visual_segment → {INFERENCE_URL}/sam3/visual_segment
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

  // Forward any api_key from the incoming query as well
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
    return NextResponse.json(
      {
        detail: `Cannot reach SAM3 inference at ${base}${targetPath}: ${message}`,
        hint: "Start the Python sidecar with `./sidecar/run.sh` or set INFERENCE_URL to Roboflow.",
      },
      { status: 502 },
    );
  }
}

export async function GET() {
  return NextResponse.json({
    ok: true,
    inferenceUrl: inferenceBaseUrl(),
    endpoints: [
      "/api/sam3/embed_image",
      "/api/sam3/visual_segment",
      "/api/sam3/concept_segment",
    ],
  });
}
