import { useRef, useEffect, useCallback } from "react";
import type { IDrawingEngine, BrushSettings } from "../engine/types";

interface DrawingCanvasProps {
  engine: IDrawingEngine;
  brush: BrushSettings;
}

/**
 * Viewport canvas that translates mouse input into paint operations on
 * the engine's hidden 4096×4096 surface.
 */
export function DrawingCanvas({ engine, brush }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);
  const lastPos = useRef<{ x: number; y: number } | null>(null);

  // Keep a mutable ref so in‑flight drag events always see the latest
  // brush settings without re‑binding every callback.
  const brushRef = useRef(brush);
  brushRef.current = brush;

  /* ---- resize the viewport canvas to fill its container ---- */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      canvas.width = rect.width * devicePixelRatio;
      canvas.height = rect.height * devicePixelRatio;
      engine.attachViewport(canvas);
    };

    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [engine]);

  /* ---- coordinate helper ---- */
  const toCanvas = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const vx = (e.clientX - rect.left) * (canvas.width / rect.width);
      const vy = (e.clientY - rect.top) * (canvas.height / rect.height);
      return engine.viewportToCanvas(vx, vy);
    },
    [engine],
  );

  /* ---- mouse handlers ---- */
  const handlePointerDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (e.button !== 0) return;
      isDrawing.current = true;
      const pos = toCanvas(e);
      if (pos) {
        engine.paintPoint(pos.x, pos.y, brushRef.current);
        lastPos.current = pos;
      }
    },
    [engine, toCanvas],
  );

  const handlePointerMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isDrawing.current) return;
      const pos = toCanvas(e);
      if (!pos) return;

      if (lastPos.current) {
        engine.paintLine(
          lastPos.current.x,
          lastPos.current.y,
          pos.x,
          pos.y,
          brushRef.current,
        );
      } else {
        engine.paintPoint(pos.x, pos.y, brushRef.current);
      }
      lastPos.current = pos;
    },
    [engine, toCanvas],
  );

  const handlePointerUp = useCallback(() => {
    isDrawing.current = false;
    lastPos.current = null;
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="drawing-canvas"
      onMouseDown={handlePointerDown}
      onMouseMove={handlePointerMove}
      onMouseUp={handlePointerUp}
      onMouseLeave={handlePointerUp}
    />
  );
}
