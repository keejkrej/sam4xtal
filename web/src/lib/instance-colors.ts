/** Distinct colors for multi-instance overlays and exported mask PNGs.

Background in exported masks is always `#000000`. Palette entries must never
be pure black so downstream can match pixels → instances via metadata.
*/

export type MaskRgb = { r: number; g: number; b: number };

export type InstanceColor = {
  fill: string;
  stroke: string;
  solid: string;
  rgb: MaskRgb;
};

function entry(r: number, g: number, b: number): InstanceColor {
  const hex =
    `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
  return {
    fill: `rgba(${r}, ${g}, ${b}, 0.38)`,
    stroke: `rgba(${Math.min(255, r + 100)}, ${Math.min(255, g + 100)}, ${Math.min(255, b + 100)}, 0.95)`,
    solid: hex,
    rgb: { r, g, b },
  };
}

/** Fixed colormap sampled for instance identity in the mask PNG. */
export const INSTANCE_COLORS: InstanceColor[] = [
  entry(34, 197, 94), // green
  entry(59, 130, 246), // blue
  entry(245, 158, 11), // amber
  entry(236, 72, 153), // pink
  entry(20, 184, 166), // teal
  entry(249, 115, 22), // orange
  entry(14, 165, 233), // sky
  entry(234, 179, 8), // yellow
  entry(168, 85, 247), // violet
  entry(244, 63, 94), // rose
  entry(132, 204, 22), // lime
  entry(6, 182, 212), // cyan
];

export function colorForInstance(index: number): InstanceColor {
  return INSTANCE_COLORS[((index % INSTANCE_COLORS.length) + INSTANCE_COLORS.length) % INSTANCE_COLORS.length];
}

export function rgbToHex(rgb: MaskRgb): string {
  return `#${[rgb.r, rgb.g, rgb.b]
    .map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0"))
    .join("")}`;
}
