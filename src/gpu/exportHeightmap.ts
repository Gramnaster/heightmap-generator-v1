import UPNG from "upng-js";

/**
 * Encode a Float32Array heightmap (0.0–1.0) into a **16‑bit grayscale PNG**
 * and trigger a browser download.
 *
 * @param heightData  One float per pixel, row‑major, values in [0, 1].
 * @param width       Image width  (e.g. 4096).
 * @param height      Image height (e.g. 4096).
 * @param filename    Download filename (default: "heightmap.png").
 */
export function downloadHeightmap16(
  heightData: Float32Array,
  width: number,
  height: number,
  filename = "heightmap.png",
): void {
  // ── 1.  Float32 → Uint16 (0‑65535) ─────────────────────────────
  const pixelCount = width * height;
  const raw = new Uint16Array(pixelCount);

  for (let i = 0; i < pixelCount; i++) {
    // Clamp then scale to full 16‑bit range.
    raw[i] = Math.round(Math.min(1, Math.max(0, heightData[i])) * 65535);
  }

  // UPNG expects an ArrayBuffer whose layout matches the channel depth.
  // For 16‑bit grayscale (ctype 0, depth 16) each pixel is 2 bytes BE.
  // We need to byte‑swap on little‑endian machines (virtually all browsers).
  const buf = new ArrayBuffer(pixelCount * 2);
  const view = new DataView(buf);
  for (let i = 0; i < pixelCount; i++) {
    view.setUint16(i * 2, raw[i], false); // big‑endian
  }

  // ── 2.  Encode via UPNG ────────────────────────────────────────
  // UPNG.encode(imgs, w, h, colorDepth)
  //   imgs : ArrayBuffer[]  — one frame
  //   colorDepth = 0 means lossless; we pass bit depth via the raw data.
  const pngArrayBuffer = UPNG.encode(
    [buf],   // single frame
    width,
    height,
    0,       // 0 = lossless (no palette quantisation)
  );

  // ── 3.  Trigger download ───────────────────────────────────────
  const blob = new Blob([pngArrayBuffer], { type: "image/png" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();

  // Clean up
  requestAnimationFrame(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}
