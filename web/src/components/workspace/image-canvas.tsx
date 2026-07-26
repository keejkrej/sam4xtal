"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { colorForInstance } from "@/lib/instance-colors";
import type { PointPrompt, SegmentInstance } from "@/lib/types";

export type ClickMode = "off" | "positive" | "negative";

type Props = {
  src: string;
  instances: SegmentInstance[];
  activeInstanceId: string | null;
  onAddPoint: (point: PointPrompt) => void;
  onRemovePoint: (index: number) => void;
  onSelectInstance?: (id: string) => void;
  clickMode?: ClickMode;
  disabled?: boolean;
};

export function ImageCanvas({
  src,
  instances,
  activeInstanceId,
  onAddPoint,
  onRemovePoint,
  onSelectInstance,
  clickMode = "off",
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
  const prompting = clickMode !== "off";
  const selectEnabled = clickMode === "off" && Boolean(onSelectInstance);

  const strokeBase = Math.max(1, natural.w / 400);
  const pointR = Math.max(4, natural.w / 120);
  const hitR = Math.max(pointR * 1.75, 14 / Math.min(scaleX, scaleY));

  function handleBackgroundClick(e: React.MouseEvent<HTMLDivElement>) {
    if (disabled || !prompting) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scaleX;
    const y = (e.clientY - rect.top) / scaleY;
    onAddPoint({ x, y, positive: clickMode === "positive" });
  }

  const active = instances.find((inst) => inst.id === activeInstanceId);
  const activeIdx = instances.findIndex((inst) => inst.id === activeInstanceId);
  const activeColor = colorForInstance(Math.max(0, activeIdx));

  return (
    <div
      ref={containerRef}
      className="relative flex h-full w-full items-center justify-center overflow-hidden bg-muted"
    >
      <div
        className={`relative ${prompting ? "cursor-crosshair" : "cursor-default"}`}
        style={{ width: display.w, height: display.h }}
        onClick={handleBackgroundClick}
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
        {/* Masks — selectable only in Off mode */}
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${natural.w} ${natural.h}`}
          preserveAspectRatio="none"
        >
          {instances.map((inst, idx) => {
            const color = colorForInstance(idx);
            const isActive = inst.id === activeInstanceId;
            return (
              <g
                key={inst.id}
                opacity={isActive ? 1 : 0.55}
                onClick={
                  selectEnabled
                    ? (e) => {
                        e.stopPropagation();
                        onSelectInstance?.(inst.id);
                      }
                    : undefined
                }
                style={{ pointerEvents: selectEnabled ? "auto" : "none" }}
                className={selectEnabled ? "cursor-pointer" : undefined}
              >
                {inst.polygons.map((poly, i) => (
                  <polygon
                    key={`${inst.id}-poly-${i}`}
                    points={poly.map(([px, py]) => `${px},${py}`).join(" ")}
                    fill={color.fill}
                    stroke={color.stroke}
                    strokeWidth={isActive ? strokeBase * 1.8 : strokeBase}
                  />
                ))}
              </g>
            );
          })}
        </svg>
        {/* Points above masks — click always removes */}
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${natural.w} ${natural.h}`}
          preserveAspectRatio="none"
        >
          {(active?.points ?? []).map((p, i) => (
            <g key={`pt-${i}`}>
              <circle
                cx={p.x}
                cy={p.y}
                r={hitR}
                fill="rgba(0,0,0,0)"
                className="cursor-pointer"
                style={{ pointerEvents: disabled ? "none" : "all" }}
                onClick={(e) => {
                  e.stopPropagation();
                  if (disabled) return;
                  onRemovePoint(i);
                }}
              />
              <circle
                cx={p.x}
                cy={p.y}
                r={pointR}
                fill={p.positive ? "#fff" : "#ef4444"}
                stroke={p.positive ? activeColor.solid : "#fff"}
                strokeWidth={Math.max(1.5, natural.w / 300)}
                className="pointer-events-none"
              />
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
