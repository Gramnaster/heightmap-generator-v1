import type { BrushSettings, IDrawingEngine } from "./types";

const CANVAS_SIZE = 4096;

/**
 * Canvas 2D–backed drawing engine.
 *
 * All painting happens on a hidden 4096×4096 `<canvas>`.  A separate
 * "viewport" canvas is kept in sync by down‑scaling the hidden surface
 * after every paint operation.
 */
export class Canvas2DEngine implements IDrawingEngine {
  private readonly hiddenCanvas: HTMLCanvasElement;
  private readonly hiddenCtx: CanvasRenderingContext2D;

  private viewportCanvas: HTMLCanvasElement | null = null;
  private viewportCtx: CanvasRenderingContext2D | null = null;

  constructor() {
    this.hiddenCanvas = document.createElement("canvas");
    this.hiddenCanvas.width = CANVAS_SIZE;
    this.hiddenCanvas.height = CANVAS_SIZE;

    const ctx = this.hiddenCanvas.getContext("2d", { willReadFrequently: false });
    if (!ctx) throw new Error("Could not obtain 2D context for hidden canvas");
    this.hiddenCtx = ctx;

    // Default background = black
    this.clear();
  }

  /* ------------------------------------------------------------------ */
  /*  Public API (IDrawingEngine)                                       */
  /* ------------------------------------------------------------------ */

  attachViewport(canvas: HTMLCanvasElement): void {
    this.viewportCanvas = canvas;
    this.viewportCtx = canvas.getContext("2d");
    this.renderViewport();
  }

  clear(): void {
    this.hiddenCtx.fillStyle = "#000000";
    this.hiddenCtx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    this.renderViewport();
  }

  paintPoint(x: number, y: number, brush: BrushSettings): void {
    this.stamp(x, y, brush);
    this.renderViewport();
  }

  paintLine(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    brush: BrushSettings,
  ): void {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const dist = Math.sqrt(dx * dx + dy * dy);

    // Spacing between dabs – tighter spacing = smoother stroke
    const spacing = Math.max(1, brush.size * 0.15);
    const steps = Math.ceil(dist / spacing);

    for (let i = 0; i <= steps; i++) {
      const t = steps === 0 ? 0 : i / steps;
      this.stamp(x0 + dx * t, y0 + dy * t, brush);
    }

    this.renderViewport();
  }

  renderViewport(): void {
    if (!this.viewportCanvas || !this.viewportCtx) return;

    const vw = this.viewportCanvas.width;
    const vh = this.viewportCanvas.height;
    const ctx = this.viewportCtx;

    ctx.clearRect(0, 0, vw, vh);

    const scale = Math.min(vw / CANVAS_SIZE, vh / CANVAS_SIZE);
    const offsetX = (vw - CANVAS_SIZE * scale) / 2;
    const offsetY = (vh - CANVAS_SIZE * scale) / 2;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(
      this.hiddenCanvas,
      offsetX,
      offsetY,
      CANVAS_SIZE * scale,
      CANVAS_SIZE * scale,
    );
  }

  viewportToCanvas(vx: number, vy: number): { x: number; y: number } | null {
    if (!this.viewportCanvas) return null;

    const vw = this.viewportCanvas.width;
    const vh = this.viewportCanvas.height;
    const scale = Math.min(vw / CANVAS_SIZE, vh / CANVAS_SIZE);
    const offsetX = (vw - CANVAS_SIZE * scale) / 2;
    const offsetY = (vh - CANVAS_SIZE * scale) / 2;

    const x = (vx - offsetX) / scale;
    const y = (vy - offsetY) / scale;

    if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) return null;
    return { x, y };
  }

  getCanvasSize(): number {
    return CANVAS_SIZE;
  }

  getHiddenCanvas(): HTMLCanvasElement {
    return this.hiddenCanvas;
  }

  dispose(): void {
    this.viewportCanvas = null;
    this.viewportCtx = null;
  }

  /* ------------------------------------------------------------------ */
  /*  Internals                                                         */
  /* ------------------------------------------------------------------ */

  /**
   * Draw a single brush dab at (x, y) on the hidden canvas.
   *
   * Uses a radial gradient when softness > 0 to produce feathered edges.
   */
  private stamp(x: number, y: number, brush: BrushSettings): void {
    const ctx = this.hiddenCtx;
    const radius = brush.size / 2;
    if (radius <= 0) return;

    const g = brush.color;
    const solid = `rgb(${g},${g},${g})`;

    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.closePath();

    if (brush.softness > 0.01) {
      const innerR = radius * (1 - brush.softness);
      const gradient = ctx.createRadialGradient(x, y, innerR, x, y, radius);
      gradient.addColorStop(0, solid);
      gradient.addColorStop(1, `rgba(${g},${g},${g},0)`);
      ctx.fillStyle = gradient;
    } else {
      ctx.fillStyle = solid;
    }

    ctx.fill();
    ctx.restore();
  }
}
