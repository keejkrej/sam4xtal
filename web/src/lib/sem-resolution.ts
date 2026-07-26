/**
 * Read SEM pixel scale (nm/px) from image file metadata when present.
 *
 * Primary: Zeiss SmartSEM private TIFF tag 34118 (0x8546) text dump, e.g.
 *   Image Pixel Size = 3.45 nm
 * Also: FEI/Thermo Fisher TIFF tag 34683 XML PixelWidth (meters).
 */

export type SemResolutionSource = "zeiss-smartsem" | "fei" | "imagej";

export type SemResolution = {
  nmPerPx: number;
  source: SemResolutionSource;
};

const TIFF_TAG_IMAGE_DESCRIPTION = 270;
const TIFF_TAG_ZEISS_SEM = 34118; // 0x8546 CZ_SEM
const TIFF_TAG_FEI = 34683; // 0x877B

const TYPE_SIZE: Record<number, number> = {
  1: 1, // BYTE
  2: 1, // ASCII
  3: 2, // SHORT
  4: 4, // LONG
  5: 8, // RATIONAL
  7: 1, // UNDEFINED
};

function isTiffName(name: string): boolean {
  return /\.tif{1,2}$/i.test(name);
}

function toNm(value: number, unit: string | undefined): number | null {
  if (!Number.isFinite(value) || value <= 0) return null;
  const u = (unit ?? "nm").toLowerCase().replace("µ", "u").replace("μ", "u");
  switch (u) {
    case "nm":
    case "nanometer":
    case "nanometers":
      return value;
    case "pm":
    case "picometer":
    case "picometers":
      return value / 1000;
    case "um":
    case "micron":
    case "microns":
    case "micrometer":
    case "micrometers":
      return value * 1000;
    case "mm":
    case "millimeter":
    case "millimeters":
      return value * 1e6;
    case "m":
    case "meter":
    case "meters":
      return value * 1e9;
    default:
      return null;
  }
}

/** Parse Zeiss / generic "Image Pixel Size = X nm" style lines. */
export function parseZeissPixelSize(text: string): number | null {
  const patterns = [
    /Image\s+Pixel\s+Size\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*([A-Za-zµμ]+)?/i,
    /Pixel\s+Size\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*([A-Za-zµμ]+)?/i,
  ];
  for (const re of patterns) {
    const m = text.match(re);
    if (!m) continue;
    const nm = toNm(Number(m[1]), m[2] ?? "nm");
    if (nm != null) return nm;
  }
  return null;
}

/** Parse FEI/Thermo Fisher XML PixelWidth (typically meters). */
export function parseFeiPixelWidth(text: string): number | null {
  const m = text.match(/<PixelWidth>\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*<\/PixelWidth>/i);
  if (!m) return null;
  // FEI stores meters; if the number is already "nm-scale" (>1e-3), treat as meters still.
  const meters = Number(m[1]);
  if (!Number.isFinite(meters) || meters <= 0) return null;
  return meters * 1e9;
}

/** ImageJ-style ImageDescription: unit=nm + spacing=… or xy spacing. */
export function parseImageJDescription(text: string): number | null {
  if (!/ImageJ/i.test(text) && !/\bunit\s*=/i.test(text)) return null;
  const unitMatch = text.match(/\bunit\s*=\s*([A-Za-zµμ]+)/i);
  const spacingMatch =
    text.match(/\bspacing\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)/i) ??
    text.match(/\bx\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)/i);
  if (!spacingMatch) return null;
  return toNm(Number(spacingMatch[1]), unitMatch?.[1] ?? "nm");
}

function decodeAsciiish(bytes: Uint8Array): string {
  // Prefer UTF-8; fall back to latin1 for proprietary dumps.
  try {
    const utf8 = new TextDecoder("utf-8", { fatal: false }).decode(bytes);
    if (!utf8.includes("\uFFFD")) return utf8;
  } catch {
    /* ignore */
  }
  let s = "";
  for (let i = 0; i < bytes.length; i++) {
    const c = bytes[i];
    if (c === 0) break;
    s += String.fromCharCode(c);
  }
  return s;
}

type TiffReader = {
  view: DataView;
  le: boolean;
};

function readU16(r: TiffReader, offset: number): number {
  return r.view.getUint16(offset, r.le);
}

function readU32(r: TiffReader, offset: number): number {
  return r.view.getUint32(offset, r.le);
}

function readTagBytes(
  r: TiffReader,
  type: number,
  count: number,
  valueOffset: number,
): Uint8Array | null {
  const unit = TYPE_SIZE[type];
  if (!unit) return null;
  const size = unit * count;
  if (size <= 0 || size > r.view.byteLength) return null;
  let start = valueOffset;
  if (size <= 4) {
    start = valueOffset;
    // Inline value sits at the entry's value field; caller passes that absolute offset.
  }
  if (start < 0 || start + size > r.view.byteLength) return null;
  return new Uint8Array(r.view.buffer, r.view.byteOffset + start, size);
}

function parseIfd(r: TiffReader, ifdOffset: number): SemResolution | null {
  if (ifdOffset <= 0 || ifdOffset + 2 > r.view.byteLength) return null;
  const numEntries = readU16(r, ifdOffset);
  if (numEntries <= 0 || numEntries > 4096) return null;

  let zeiss: string | null = null;
  let fei: string | null = null;
  let description: string | null = null;

  for (let i = 0; i < numEntries; i++) {
    const entry = ifdOffset + 2 + i * 12;
    if (entry + 12 > r.view.byteLength) break;
    const tag = readU16(r, entry);
    const type = readU16(r, entry + 2);
    const count = readU32(r, entry + 4);
    const unit = TYPE_SIZE[type];
    if (!unit) continue;
    const size = unit * count;
    const valueField = entry + 8;
    const dataOffset = size <= 4 ? valueField : readU32(r, valueField);
    if (
      tag !== TIFF_TAG_ZEISS_SEM &&
      tag !== TIFF_TAG_FEI &&
      tag !== TIFF_TAG_IMAGE_DESCRIPTION
    ) {
      continue;
    }
    const bytes = readTagBytes(r, type, count, dataOffset);
    if (!bytes) continue;
    const text = decodeAsciiish(bytes);
    if (tag === TIFF_TAG_ZEISS_SEM) zeiss = text;
    else if (tag === TIFF_TAG_FEI) fei = text;
    else description = text;
  }

  if (zeiss) {
    const nm = parseZeissPixelSize(zeiss);
    if (nm != null) return { nmPerPx: nm, source: "zeiss-smartsem" };
  }
  if (fei) {
    const nm = parseFeiPixelWidth(fei);
    if (nm != null) return { nmPerPx: nm, source: "fei" };
  }
  if (description) {
    const nm = parseImageJDescription(description);
    if (nm != null) return { nmPerPx: nm, source: "imagej" };
  }
  return null;
}

/** Extract nm/px from a TIFF ArrayBuffer, or null if unavailable. */
export function readSemResolutionFromBuffer(
  buffer: ArrayBuffer,
): SemResolution | null {
  if (buffer.byteLength < 8) return null;
  const view = new DataView(buffer);
  const b0 = view.getUint8(0);
  const b1 = view.getUint8(1);
  const le = b0 === 0x49 && b1 === 0x49;
  const be = b0 === 0x4d && b1 === 0x4d;
  if (!le && !be) return null;
  const r: TiffReader = { view, le };
  const magic = readU16(r, 2);
  if (magic !== 42) return null; // classic TIFF only (not BigTIFF)

  let ifdOffset = readU32(r, 4);
  // Walk a few IFDs; SEM metadata is almost always on the first.
  for (let n = 0; n < 8 && ifdOffset; n++) {
    const found = parseIfd(r, ifdOffset);
    if (found) return found;
    const numEntries = readU16(r, ifdOffset);
    const nextPos = ifdOffset + 2 + numEntries * 12;
    if (nextPos + 4 > view.byteLength) break;
    ifdOffset = readU32(r, nextPos);
  }
  return null;
}

export async function readSemResolutionFromFile(
  file: File,
): Promise<SemResolution | null> {
  if (!isTiffName(file.name) && file.type !== "image/tiff" && file.type !== "image/tif") {
    return null;
  }
  try {
    const buffer = await file.arrayBuffer();
    return readSemResolutionFromBuffer(buffer);
  } catch {
    return null;
  }
}

export function formatNmPerPx(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "";
  // Keep enough precision for fine SEM scales without trailing junk.
  const s = n.toPrecision(8);
  return String(Number(s));
}

export function resolutionSourceLabel(source: SemResolutionSource): string {
  switch (source) {
    case "zeiss-smartsem":
      return "from TIFF (Zeiss SmartSEM)";
    case "fei":
      return "from TIFF (FEI)";
    case "imagej":
      return "from TIFF (ImageJ)";
  }
}
