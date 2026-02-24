import { initWebGPU } from "./initWebGPU";
import { extractHeightmap } from "./extractHeightmap";

/* ------------------------------------------------------------------ */
/*  Compute shader (WGSL)                                             */
/* ------------------------------------------------------------------ */

/**
 * Terrain‑generation compute shader  — v2 "Layered Math" engine.
 *
 * Reads the painted heightmap (0.0–1.0 grayscale) and processes it
 * through four geological layers:
 *
 *   1. Domain Warp     – distort UVs with noise so brush strokes get
 *                         organic, tectonic‑style flow lines.
 *   2. Continental Base – low‑frequency FBM + power‑curve exponent to
 *                         create vast flat plains with sudden elevation
 *                         changes  (fixes the "Perlin Pillow" effect).
 *   3. Mountain Ridges  – ridged multifractal noise applied only where
 *                         the user painted white, producing sharp peaks
 *                         instead of round domes.
 *   4. Final Blend      – smooth interpolation between layers governed
 *                         by the painted value (black = flat, gray =
 *                         rolling hills, white = towering mountains).
 *
 * Bindings:
 *   @group(0) @binding(0)  params        – uniform { width, height }
 *   @group(0) @binding(1)  heightmapIn   – read‑only  storage<f32[]>
 *   @group(0) @binding(2)  heightmapOut  – read‑write storage<f32[]>
 */
const COMPUTE_SHADER_SRC = /* wgsl */ `

/* ---- uniforms ---- */
struct Params {
  width  : u32,
  height : u32,
};
@group(0) @binding(0) var<uniform>             params       : Params;
@group(0) @binding(1) var<storage, read>       heightmapIn  : array<f32>;
@group(0) @binding(2) var<storage, read_write> heightmapOut : array<f32>;

/* ================================================================== */
/*  Hash / noise primitives                                           */
/* ================================================================== */

// 2→1 hash (good distribution, cheap ALU).
fn hash2(p : vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q = q + dot(q, q + 45.32);
  return fract(q.x * q.y);
}

// 2→2 hash – needed for domain warping (two independent offsets).
fn hash22(p : vec2f) -> vec2f {
  let a = hash2(p);
  let b = hash2(p + vec2f(37.0, 91.0));
  return vec2f(a, b);
}

// 2D value noise with quintic interpolation (C2‑continuous).
fn valueNoise(p : vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

  let a = hash2(i + vec2f(0.0, 0.0));
  let b = hash2(i + vec2f(1.0, 0.0));
  let c = hash2(i + vec2f(0.0, 1.0));
  let d = hash2(i + vec2f(1.0, 1.0));

  return mix(mix(a, b, u.x),
             mix(c, d, u.x), u.y);
}

// 30°‑rotation matrix applied per‑octave to break grid alignment.
const ROT : mat2x2f = mat2x2f(
  vec2f( 0.866, 0.5),
  vec2f(-0.5,   0.866)
);

/* ================================================================== */
/*  Layer 1 – Domain Warping                                          */
/*                                                                    */
/*  Uses noise to bend the UV coordinates before they enter the       */
/*  terrain generators.  Creates sweeping tectonic curves that make   */
/*  brush‑stroke edges look geological instead of perfectly round.    */
/* ================================================================== */

fn domainWarp(p : vec2f) -> vec2f {
  // Two independent noise lookups (offset seeds to decorrelate).
  let warpStrength = 0.35;  // how far UVs get pushed (in UV‑space)
  let warpFreq     = 2.0;   // scale of the warping pattern

  let wx = valueNoise(p * warpFreq)               * 2.0 - 1.0;
  let wy = valueNoise(p * warpFreq + vec2f(50.0, 50.0)) * 2.0 - 1.0;

  return p + vec2f(wx, wy) * warpStrength;
}

/* ================================================================== */
/*  Layer 2 – Continental Base  (FBM + power‑curve)                   */
/*                                                                    */
/*  Low‑frequency rolling noise run through pow(val, exponent) so     */
/*  low areas flatten to near‑zero (plains / ocean beds) while high   */
/*  areas remain tall (continental shelves / plateaus).                */
/* ================================================================== */

fn fbm(pos : vec2f, octaves : i32) -> f32 {
  var value = 0.0;
  var amp   = 0.5;
  var freq  = 1.0;
  var p     = pos;

  for (var i = 0; i < octaves; i = i + 1) {
    value = value + amp * valueNoise(p * freq);
    freq  = freq * 2.0;
    amp   = amp  * 0.5;
    p     = ROT * p;
  }
  return value;
}

fn continentalNoise(p : vec2f) -> f32 {
  // Low‑frequency base (3 octaves — we only want broad shapes).
  let raw = fbm(p * 1.5, 3);
  // Power curve: flattens valleys, sharpens elevation transitions.
  return pow(raw, 2.5);
}

/* ================================================================== */
/*  Layer 3 – Mountain Ridges  (ridged multifractal)                  */
/*                                                                    */
/*  |noise| flipped + squared creates sharp ridge‑lines.  Octave     */
/*  weighting feeds the previous signal into the next amplitude,      */
/*  concentrating detail around existing peaks.                       */
/* ================================================================== */

fn ridgedNoise(pos : vec2f, octaves : i32) -> f32 {
  var value  = 0.0;
  var amp    = 0.6;
  var freq   = 1.0;
  var weight = 1.0;
  var p      = pos;

  for (var i = 0; i < octaves; i = i + 1) {
    var n = valueNoise(p * freq);
    // Fold to V‑shape, flip to ridge, sharpen.
    n = 1.0 - abs(n * 2.0 - 1.0);
    n = n * n;
    // Weight by previous signal – concentrates detail on ridges.
    n = n * weight;
    weight = clamp(n * 1.2, 0.0, 1.0);

    value = value + n * amp;
    freq  = freq * 2.2;   // slightly non‑integer ratio → less tiling
    amp   = amp  * 0.45;
    p     = ROT * p;
  }
  return value;
}

/* ================================================================== */
/*  Layer 4 – Detail FBM  (adds small‑scale roughness everywhere)     */
/* ================================================================== */

fn detailNoise(p : vec2f) -> f32 {
  // High‑frequency fine detail (4 octaves).
  return fbm(p * 6.0, 4);
}

/* ================================================================== */
/*  Main compute entry                                                */
/* ================================================================== */

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let px = gid.x;
  let py = gid.y;
  let w  = params.width;
  let h  = params.height;
  if (px >= w || py >= h) {
    return;
  }

  let idx = py * w + px;
  let painted = heightmapIn[idx];

  // ── UV coordinates ──────────────────────────────────────────────
  let uv = vec2f(f32(px), f32(py)) / vec2f(f32(w), f32(h));

  // ── Layer 1: Domain Warp ────────────────────────────────────────
  // Distort the coordinates so brush edges become organic flow lines.
  let warped = domainWarp(uv * 8.0);

  // ── Layer 2: Continental Base ───────────────────────────────────
  // Broad, flat plains with pow(2.5) curve.  Applied everywhere
  // the user paints above black.
  let continental = continentalNoise(warped);

  // ── Layer 3: Mountain Ridges ────────────────────────────────────
  // Sharp ridged‑multifractal.  Only engaged where painted → white.
  let ridges = ridgedNoise(warped * 1.3, 7);

  // ── Layer 4: Fine Detail ────────────────────────────────────────
  // Subtle roughness layered on top of everything.
  let detail = detailNoise(warped);

  // ── Blend zones (governed by painted value) ─────────────────────
  //
  //  painted ≈ 0.0  →  flat ground (near zero output)
  //  painted ≈ 0.5  →  rolling continental hills + detail
  //  painted ≈ 1.0  →  towering ridged mountains + detail
  //
  // hillWeight: bell curve peaking at painted ≈ 0.5
  let hillWeight = smoothstep(0.05, 0.30, painted)
                 * smoothstep(0.90, 0.55, painted);

  // mtnWeight: ramps in above painted ≈ 0.5
  let mtnWeight = smoothstep(0.40, 0.80, painted);

  // detailWeight: fades in whenever anything is painted
  let detailWeight = smoothstep(0.05, 0.25, painted) * 0.06;

  // ── Combine layers ──────────────────────────────────────────────
  // Continental base: max ≈ 0.4 (after pow‑curve, values are small).
  let hillHeight = continental * 0.5;

  // Mountains: ridged × painted value — brighter paint = taller.
  let mtnHeight = ridges * painted;

  // Detail: subtle everywhere there is terrain.
  let detailHeight = detail;

  var height = hillWeight  * hillHeight
             + mtnWeight   * mtnHeight
             + detailWeight * detailHeight;

  // Final power‑curve on the combined result: pushes remaining
  // low values even flatter while preserving peaks.
  height = pow(clamp(height, 0.0, 1.0), 1.3);

  heightmapOut[idx] = clamp(height, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/*  Cached GPU resources                                              */
/* ------------------------------------------------------------------ */

let gpuDevice: GPUDevice | null = null;
let computePipeline: GPUComputePipeline | null = null;
let bindGroupLayout: GPUBindGroupLayout | null = null;

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

export interface HeightmapGPUResult {
  /** The GPU device used (handy for further rendering work). */
  device: GPUDevice;
  /** The output storage buffer living on the GPU. */
  outputBuffer: GPUBuffer;
  /** Width × height of the heightmap. */
  pixelCount: number;
}

/**
 * Run the heightmap through the WebGPU compute pipeline.
 *
 * 1. Extracts ImageData from the hidden canvas.
 * 2. Normalises to Float32 (0.0 – 1.0).
 * 3. Uploads to a GPU storage buffer.
 * 4. Dispatches a compute shader.
 * 5. Returns a handle to the output buffer on the GPU.
 */
export async function runHeightmapCompute(
  hiddenCanvas: HTMLCanvasElement,
): Promise<HeightmapGPUResult> {
  /* ---- 1.  Ensure we have a device + pipeline ---- */
  if (!gpuDevice || !computePipeline || !bindGroupLayout) {
    const { device } = await initWebGPU();
    gpuDevice = device;

    bindGroupLayout = device.createBindGroupLayout({
      label: "heightmap-bind-group-layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const shaderModule = device.createShaderModule({
      label: "heightmap-compute-shader",
      code: COMPUTE_SHADER_SRC,
    });

    computePipeline = device.createComputePipeline({
      label: "heightmap-compute-pipeline",
      layout: device.createPipelineLayout({
        label: "heightmap-pipeline-layout",
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
  }

  const device = gpuDevice;

  /* ---- 2.  Extract + normalise heightmap ---- */
  const heightmap = extractHeightmap(hiddenCanvas);
  const pixelCount = heightmap.length;
  const byteSize = heightmap.byteLength; // pixelCount * 4

  const canvasWidth = hiddenCanvas.width;
  const canvasHeight = hiddenCanvas.height;

  /* ---- 3.  Create GPU buffers ---- */

  // Uniform buffer: { width: u32, height: u32 }  (8 bytes, 16‑byte aligned)
  const uniformData = new Uint32Array([canvasWidth, canvasHeight]);
  const uniformBuffer = device.createBuffer({
    label: "heightmap-params",
    size: 16, // minimum uniform buffer alignment
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  const inputBuffer = device.createBuffer({
    label: "heightmap-input",
    size: byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const outputBuffer = device.createBuffer({
    label: "heightmap-output",
    size: byteSize,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  // Upload the heightmap data
  device.queue.writeBuffer(inputBuffer, 0, heightmap.buffer);

  /* ---- 4.  Create bind group + dispatch ---- */
  const bindGroup = device.createBindGroup({
    label: "heightmap-bind-group",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
    ],
  });

  const wgSize = 16;
  const workgroupsX = Math.ceil(canvasWidth / wgSize);
  const workgroupsY = Math.ceil(canvasHeight / wgSize);

  const encoder = device.createCommandEncoder({
    label: "heightmap-compute-encoder",
  });
  const pass = encoder.beginComputePass({ label: "heightmap-compute-pass" });
  pass.setPipeline(computePipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupsX, workgroupsY);
  pass.end();

  device.queue.submit([encoder.finish()]);

  // Wait for GPU to finish
  await device.queue.onSubmittedWorkDone();

  /* ---- 5.  Clean up transient buffers (output stays alive) ---- */
  inputBuffer.destroy();
  uniformBuffer.destroy();

  console.log(
    `[HeightmapCompute] Dispatched ${workgroupsX}×${workgroupsY} workgroups ` +
      `(${pixelCount} pixels, ${(byteSize / 1024 / 1024).toFixed(1)} MB).`,
  );

  return { device, outputBuffer, pixelCount };
}

/**
 * Read the output buffer back to the CPU (for debugging / validation).
 * Prefer keeping data on the GPU for rendering in production.
 */
export async function readbackOutputBuffer(
  device: GPUDevice,
  outputBuffer: GPUBuffer,
  pixelCount: number,
): Promise<Float32Array> {
  const byteSize = pixelCount * 4;

  const staging = device.createBuffer({
    label: "heightmap-readback-staging",
    size: byteSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(outputBuffer, 0, staging, 0, byteSize);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();

  return result;
}
