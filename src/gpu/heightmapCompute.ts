import { initWebGPU } from "./initWebGPU";
import { extractHeightmap } from "./extractHeightmap";

/* ------------------------------------------------------------------ */
/*  Compute shader (WGSL)                                             */
/* ------------------------------------------------------------------ */

/**
 * Terrain‑generation compute shader.
 *
 * Reads the painted heightmap (0.0–1.0 grayscale) and produces
 * terrain using two noise regimes blended by the painted value:
 *
 *   • Mid‑gray (≈0.5)  → smooth FBM hills,  clamped to max 0.5
 *   • White    (≈1.0)  → sharp ridged‑noise mountains, scaled by
 *                         the painted value so brighter = taller.
 *
 * Values near black (≈0.0) pass through as‑is (flat ground).
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

// Fast 2→1 hash (based on Hugo Elias / integer‑noise patterns).
fn hash2(p : vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q = q + dot(q, q + 45.32);
  return fract(q.x * q.y);
}

// Gradient‑style 2D value noise with quintic interpolation.
fn valueNoise(p : vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  // Quintic Hermite curve for smooth second‑derivative continuity.
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

  let a = hash2(i + vec2f(0.0, 0.0));
  let b = hash2(i + vec2f(1.0, 0.0));
  let c = hash2(i + vec2f(0.0, 1.0));
  let d = hash2(i + vec2f(1.0, 1.0));

  return mix(mix(a, b, u.x),
             mix(c, d, u.x), u.y);
}

/* ================================================================== */
/*  FBM – Fractal Brownian Motion  (smooth, rolling hills)            */
/* ================================================================== */

fn fbm(pos : vec2f, octaves : i32) -> f32 {
  var value = 0.0;
  var amp   = 0.5;
  var freq  = 1.0;
  var p     = pos;

  for (var i = 0; i < octaves; i = i + 1) {
    value = value + amp * valueNoise(p * freq);
    freq  = freq  * 2.0;
    amp   = amp   * 0.5;
    // Rotate slightly each octave to break axis‑alignment.
    p = vec2f(p.x * 0.866 - p.y * 0.5,
              p.x * 0.5   + p.y * 0.866);
  }
  return value;
}

/* ================================================================== */
/*  Ridged noise  (sharp peaks, craggy mountains)                     */
/* ================================================================== */

fn ridgedNoise(pos : vec2f, octaves : i32) -> f32 {
  var value  = 0.0;
  var amp    = 0.5;
  var freq   = 1.0;
  var weight = 1.0;
  var p      = pos;

  for (var i = 0; i < octaves; i = i + 1) {
    var n = valueNoise(p * freq);
    // Fold the noise to create ridges.
    n = 1.0 - abs(n * 2.0 - 1.0);
    // Square it to sharpen the ridges.
    n = n * n;
    // Weight successive octaves by previous signal.
    n = n * weight;
    weight = clamp(n, 0.0, 1.0);

    value = value + n * amp;
    freq  = freq * 2.0;
    amp   = amp  * 0.5;
    p = vec2f(p.x * 0.866 - p.y * 0.5,
              p.x * 0.5   + p.y * 0.866);
  }
  return value;
}

/* ================================================================== */
/*  Main compute entry                                                */
/* ================================================================== */

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let idx = gid.x;
  let total = arrayLength(&heightmapIn);
  if (idx >= total) {
    return;
  }

  let painted = heightmapIn[idx];

  // ── Convert flat index → 2D UV ──────────────────────────────────
  let w  = params.width;
  let px = idx % w;
  let py = idx / w;
  // UV in 0..1, then scale to a nice noise frequency.
  let uv = vec2f(f32(px), f32(py)) / vec2f(f32(w), f32(params.height));
  let noiseCoord = uv * 8.0;   // tile frequency – tweak to taste

  // ── Noise evaluation ────────────────────────────────────────────
  let hillNoise     = fbm(noiseCoord, 6);                 // 0..~1
  let mountainNoise = ridgedNoise(noiseCoord * 1.2, 6);   // 0..~1

  // ── Blend zones ─────────────────────────────────────────────────
  // Black  (0.0) → flat ground, keep near 0.
  // Mid    (0.5) → rolling hills  (FBM), max height 0.5.
  // White  (1.0) → towering mountains (ridged × painted).

  // How much "hill" influence (peaks around painted == 0.5).
  let hillWeight = smoothstep(0.05, 0.35, painted) *
                   smoothstep(0.95, 0.65, painted);

  // How much "mountain" influence (ramps up toward painted == 1.0).
  let mtnWeight = smoothstep(0.45, 0.85, painted);

  // Hills: FBM scaled so max stays ≤ 0.5.
  let hillHeight = hillNoise * 0.5;

  // Mountains: ridged noise scaled by the painted value itself,
  // so brighter strokes → taller peaks.
  let mtnHeight = mountainNoise * painted;

  // Blend the two contributions.
  let height = hillWeight * hillHeight + mtnWeight * mtnHeight;

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

  const workgroupSize = 256;
  const workgroupCount = Math.ceil(pixelCount / workgroupSize);

  const encoder = device.createCommandEncoder({
    label: "heightmap-compute-encoder",
  });
  const pass = encoder.beginComputePass({ label: "heightmap-compute-pass" });
  pass.setPipeline(computePipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupCount);
  pass.end();

  device.queue.submit([encoder.finish()]);

  // Wait for GPU to finish
  await device.queue.onSubmittedWorkDone();

  /* ---- 5.  Clean up transient buffers (output stays alive) ---- */
  inputBuffer.destroy();
  uniformBuffer.destroy();

  console.log(
    `[HeightmapCompute] Dispatched ${workgroupCount} workgroups ` +
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
