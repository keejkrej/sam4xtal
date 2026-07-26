"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { colorForInstance } from "@/lib/instance-colors";
import type { PointPrompt, SegmentInstance } from "@/lib/types";

type Props = {
  src: string;
  instances: SegmentInstance[];
  activeInstanceId: string | null;
  onAddPoint: (point: PointPrompt) => void;
  onSelectInstance?: (id: string) => void;
  negativeMode?: boolean;
  disabled?: boolean;
};

export function ImageCanvas({
  src,
  instances,
  activeInstanceId,
  onAddPoint,
  onSelectInstance,
  negativeMode = false,
  disabled = false,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [natural, setNatural] = useState({ w: 1, h: 1 });
  const [display, setDisplay] = useState({ w: 1, h: 1 });

  const measure = useCallback(() => {
    const img = imgRef.current;
    const box = containerRef.current;
    if (!img || !box) return;
    const nw = img.naturalWidth || 1;
    const nh = img.naturalHeight || 1;
    setNatural({ w: nw, h: nh });

    const maxW = box.clientWidth;
    const maxH = box.clientHeight;
    const scale = Math.min(maxW / nw, maxH / nh);
    setDisplay({ w: Math.max(1, nw * scale), h: Math.max(1, nh * scale) });
  }, []);

  useEffect(() => {
    measure();
    const ro = new ResizeObserver(measure);
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [src, measure]);

  const scaleX = display.w / natural.w;
  const scaleY = display.h / natural.h;

  function handleClick(e: React.MouseEvent<HTMLDivElement>) {
    if (disabled) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const dx = e.clientX - rect.left;
    const dy = e.clientY - rect.top;
    const x = dx / scaleX;
    const y = dy / scaleY;
    onAddPoint({ x, y, positive: !negativeMode });
  }

  const strokeBase = Math.max(1, natural.w / 400);
  const pointR = Math.max(4, natural.w / 120);

  return (
    <div
      ref={containerRef}
      className="relative flex h-full w-full items-center justify-center overflow-hidden bg-neutral-950"
    >
      <div
        className="relative cursor-crosshair"
        style={{ width: display.w, height: display.h }}
        onClick={handleClick}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imgRef}
          src={src}
          alt=""
          className="block h-full w-full select-none"
          draggable={false}
          onLoad={measure}
        />
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${natural.w} ${natural.h}`}
          preserveAspectRatio="none"
        >
          {instances.map((inst, idx) => {
            const color = colorForInstance(idx);
            const active = inst.id === activeInstanceId;
            return (
              <g
                key={inst.id}
                opacity={active ? 1 : 0.55}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectInstance?.(inst.id);
                }}
                style={{ pointerEvents: onSelectInstance ? "auto" : "none" }}
                className={onSelectInstance ? "cursor-pointer" : undefined}
              >
                {inst.polygons.map((poly, i) => (
                  <polygon
                    key={`${inst.id}-poly-${i}`}
                    points={poly.map(([x, y]) => `${x},${y}`).join(" ")}
                    fill={color.fill}
                    stroke={color.stroke}
                    strokeWidth={active ? strokeBase * 1.8 : strokeBase}
                  />
                ))}
                {(active ? inst.points : []).map((p, i) => (
                  <circle
                    key={`${inst.id}-pt-${i}`}
                    cx={p.x}
                    cy={p.y}
                    r={pointR}
                    fill={p.positive ? "#fff" : "#ef4444"}
                    stroke={p.positive ? color.solid : "#fff"}
                    strokeWidth={Math.max(1.5, natural.w / 300)}
                  />
                ))}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
