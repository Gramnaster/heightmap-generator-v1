/** Shared types for the drawing engine layer. */

export interface BrushSettings {
  /** Brush diameter in canvas pixels (on the 4096×4096 surface). */
  size: number;
  /** 0 = hard edge, 1 = fully feathered from centre to rim. */
  softness: number;
  /** Grayscale value 0‑255. */
  color: number;
}

/**
 * Abstract drawing‑engine contract.
 *
 * Implement this interface for each backend (Canvas 2D today, WebGPU
 * tomorrow) so the React layer stays unchanged.
 */
export interface IDrawingEngine {
  /** Hook the engine up to the on‑screen viewport canvas. */
  attachViewport(canvas: HTMLCanvasElement): void;

  /** Stamp a single brush dab. */
  paintPoint(x: number, y: number, brush: BrushSettings): void;

  /** Interpolate stamps along a line segment. */
  paintLine(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    brush: BrushSettings,
  ): void;

  /** Fill the entire surface with black. */
  clear(): void;

  /** Blit the hidden surface to the viewport. */
  renderViewport(): void;

  /** Map a viewport‑pixel coordinate → hidden‑canvas coordinate. */
  viewportToCanvas(vx: number, vy: number): { x: number; y: number } | null;

  /** Side length of the square hidden canvas. */
  getCanvasSize(): number;

  /** Return the underlying hidden canvas (useful for export). */
  getHiddenCanvas(): HTMLCanvasElement;

  /** Release resources. */
  dispose(): void;
}
