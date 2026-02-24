/**
 * Extract the grayscale heightmap from a 2D canvas and return it as a
 * normalised Float32Array (one float per pixel, range 0.0 – 1.0).
 *
 * Only the red channel is sampled because the canvas stores grayscale
 * values (R === G === B).
 */
export function extractHeightmap(canvas: HTMLCanvasElement): Float32Array {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Cannot get 2D context from hidden canvas");

  const w = canvas.width;
  const h = canvas.height;
  const imageData = ctx.getImageData(0, 0, w, h);
  const rgba = imageData.data; // Uint8ClampedArray, length = w * h * 4

  const pixelCount = w * h;
  const heightmap = new Float32Array(pixelCount);

  for (let i = 0; i < pixelCount; i++) {
    // Take the red channel (index 0 of each RGBA quad) and normalise.
    heightmap[i] = rgba[i * 4] / 255;
  }

  return heightmap;
}
