import { initWebGPU } from "./initWebGPU";
import { extractHeightmap } from "./extractHeightmap";

/* ------------------------------------------------------------------ */
/*  Refine shader (WGSL)                                              */
/* ------------------------------------------------------------------ */

/**
 * Image‑processing compute shader that cleans up a hand‑painted
 * heightmap so it looks like a realistic landmass:
 *
 *   1.  Gaussian blur (7×7)     – smooths harsh brush edges.
 *   2.  Noise‑based edge warp   – perturbs the boundary between
 *                                  land (bright) and sea (dark) so
 *                                  coastlines look organic.
 *   3.  Contrast S‑curve        – pushes values toward black or white
 *                                  for a crisp monochrome result
 *                                  while preserving mid‑tone gradients
 *                                  needed for mountain slopes.
 *   4.  Mountain sharpening     – enhances local contrast on brighter
 *                                  pixels so mountain ridges pop.
 *
 * Bindings: same layout as the terrain shader.
 *   @group(0) @binding(0)  params        – uniform { width, height }
 *   @group(0) @binding(1)  heightmapIn   – read‑only  storage<f32[]>
 *   @group(0) @binding(2)  heightmapOut  – read‑write storage<f32[]>
 */
const REFINE_SHADER_SRC = /* wgsl */ `

struct Params {
  width  : u32,
  height : u32,
};
@group(0) @binding(0) var<uniform>             params       : Params;
@group(0) @binding(1) var<storage, read>       heightmapIn  : array<f32>;
@group(0) @binding(2) var<storage, read_write> heightmapOut : array<f32>;

/* ================================================================== */
/*  Utility                                                           */
/* ================================================================== */

fn sampleClamped(x : i32, y : i32) -> f32 {
  let cx = clamp(x, 0, i32(params.width)  - 1);
  let cy = clamp(y, 0, i32(params.height) - 1);
  return heightmapIn[u32(cy) * params.width + u32(cx)];
}

// Hash for noise perturbation.
fn hash2(p : vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q = q + dot(q, q + 45.32);
  return fract(q.x * q.y);
}

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

/* ================================================================== */
/*  Gaussian blur 7×7 (σ ≈ 2.0)                                      */
/*                                                                    */
/*  Pre‑computed kernel weights (normalised to sum to 1).             */
/* ================================================================== */

fn gaussianBlur(px : i32, py : i32) -> f32 {
  // Kernel radius = 3  →  7×7 tap neighbourhood.
  // Weights computed from exp( ‑(x²+y²)/(2·σ²) ) with σ = 2.0.
  // We compute directly in 2D (49 taps is fine at @workgroup_size 16×16).
  var total  = 0.0;
  var weight = 0.0;

  for (var dy = -3; dy <= 3; dy = dy + 1) {
    for (var dx = -3; dx <= 3; dx = dx + 1) {
      let d2 = f32(dx * dx + dy * dy);
      let w  = exp(-d2 / 8.0);          // 2·σ² = 8.0
      total  = total + sampleClamped(px + dx, py + dy) * w;
      weight = weight + w;
    }
  }
  return total / weight;
}

/* ================================================================== */
/*  Noise‑based edge perturbation                                     */
/*                                                                    */
/*  Detects edges (gradient magnitude) and shifts the sampling coord  */
/*  by a noise offset.  This breaks up perfectly round brush edges    */
/*  into natural coastline‑like contours.                             */
/* ================================================================== */

fn gradientMagnitude(px : i32, py : i32) -> f32 {
  let l = sampleClamped(px - 1, py);
  let r = sampleClamped(px + 1, py);
  let t = sampleClamped(px, py - 1);
  let b = sampleClamped(px, py + 1);
  let gx = r - l;
  let gy = b - t;
  return sqrt(gx * gx + gy * gy);
}

fn perturbedSample(px : i32, py : i32) -> f32 {
  let uv = vec2f(f32(px), f32(py)) / vec2f(f32(params.width), f32(params.height));
  let edge = gradientMagnitude(px, py);

  // Noise‑based 2D offset, strength proportional to edge strength.
  let noiseFreq = 12.0;
  let maxShift  = 8.0;   // max pixel displacement at sharpest edges
  let nx = (valueNoise(uv * noiseFreq)                          * 2.0 - 1.0) * maxShift * edge;
  let ny = (valueNoise(uv * noiseFreq + vec2f(73.1, 19.7))     * 2.0 - 1.0) * maxShift * edge;

  return sampleClamped(px + i32(round(nx)), py + i32(round(ny)));
}

/* ================================================================== */
/*  Contrast S‑curve (sigmoid)                                        */
/*                                                                    */
/*  Maps 0→0, 0.5→0.5, 1→1 but with a steep mid‑section so values    */
/*  snap toward black or white.  The steepness parameter controls how */
/*  aggressive the snap is.                                           */
/* ================================================================== */

fn contrastCurve(v : f32, steepness : f32) -> f32 {
  // Attempt a standard sigmoid remap centred at 0.5:
  //   out = 1 / (1 + exp( ‑steepness * (v ‑ 0.5) ))
  // Then re‑normalise so 0→0 and 1→1.
  let raw  = 1.0 / (1.0 + exp(-steepness * (v - 0.5)));
  let low  = 1.0 / (1.0 + exp(-steepness * (0.0 - 0.5)));
  let high = 1.0 / (1.0 + exp(-steepness * (1.0 - 0.5)));
  return (raw - low) / (high - low);
}

/* ================================================================== */
/*  Mountain sharpening (unsharp‑mask style)                          */
/*                                                                    */
/*  Computes (original ‑ blurred) to get detail, then adds it back    */
/*  weighted by brightness (so only mountains get sharpened).         */
/* ================================================================== */

fn sharpenMountains(original : f32, blurred : f32) -> f32 {
  let detail    = original - blurred;
  // Only sharpen bright areas (mountains).  lerp strength by value.
  let strength  = smoothstep(0.35, 0.75, original) * 0.6;
  return original + detail * strength;
}

/* ================================================================== */
/*  Main                                                              */
/* ================================================================== */

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let px = i32(gid.x);
  let py = i32(gid.y);
  let w  = i32(params.width);
  let h  = i32(params.height);
  if (px >= w || py >= h) {
    return;
  }

  let idx = u32(py) * params.width + u32(px);
  let original = heightmapIn[idx];

  // ── Step 1: Gaussian blur ─────────────────────────────────────
  let blurred = gaussianBlur(px, py);

  // ── Step 2: Noise‑perturbed edge sampling ─────────────────────
  // Blend: mostly blurred, but replace edge regions with perturbed
  // samples so coastlines get organic contours.
  let edge = gradientMagnitude(px, py);
  let perturbed = perturbedSample(px, py);
  // Mix: flat areas get full blur, edge areas get perturbed values.
  let edgeMix = smoothstep(0.02, 0.15, edge);
  let smoothed = mix(blurred, perturbed, edgeMix * 0.7);

  // ── Step 3: Mountain sharpening ───────────────────────────────
  let sharpened = sharpenMountains(smoothed, blurred);

  // ── Step 4: Contrast S‑curve push toward B&W ─────────────────
  let contrasted = contrastCurve(sharpened, 8.0);

  heightmapOut[idx] = clamp(contrasted, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/*  Cached GPU resources (separate from the terrain pipeline)         */
/* ------------------------------------------------------------------ */

let gpuDevice: GPUDevice | null = null;
let refinePipeline: GPUComputePipeline | null = null;
let bindGroupLayout: GPUBindGroupLayout | null = null;

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

/**
 * Run the refine compute shader on the painted heightmap.
 *
 * Returns a Float32Array (CPU) of the refined image so it can be
 * written back to the hidden canvas.
 */
export async function runRefineCompute(
  hiddenCanvas: HTMLCanvasElement,
): Promise<Float32Array> {
  /* ---- 1.  Ensure device + pipeline ---- */
  if (!gpuDevice || !refinePipeline || !bindGroupLayout) {
    const { device } = await initWebGPU();
    gpuDevice = device;

    bindGroupLayout = device.createBindGroupLayout({
      label: "refine-bind-group-layout",
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
      label: "refine-compute-shader",
      code: REFINE_SHADER_SRC,
    });

    refinePipeline = device.createComputePipeline({
      label: "refine-compute-pipeline",
      layout: device.createPipelineLayout({
        label: "refine-pipeline-layout",
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
  const byteSize = heightmap.byteLength;

  const canvasWidth = hiddenCanvas.width;
  const canvasHeight = hiddenCanvas.height;

  /* ---- 3.  Create GPU buffers ---- */
  const uniformData = new Uint32Array([canvasWidth, canvasHeight]);
  const uniformBuffer = device.createBuffer({
    label: "refine-params",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  const inputBuffer = device.createBuffer({
    label: "refine-input",
    size: byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const outputBuffer = device.createBuffer({
    label: "refine-output",
    size: byteSize,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(inputBuffer, 0, heightmap.buffer);

  /* ---- 4.  Dispatch ---- */
  const bindGroup = device.createBindGroup({
    label: "refine-bind-group",
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

  const encoder = device.createCommandEncoder({ label: "refine-encoder" });
  const pass = encoder.beginComputePass({ label: "refine-pass" });
  pass.setPipeline(refinePipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupsX, workgroupsY);
  pass.end();

  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  /* ---- 5.  Readback ---- */
  const staging = device.createBuffer({
    label: "refine-readback-staging",
    size: byteSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const copyEncoder = device.createCommandEncoder();
  copyEncoder.copyBufferToBuffer(outputBuffer, 0, staging, 0, byteSize);
  device.queue.submit([copyEncoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();

  /* ---- 6.  Cleanup ---- */
  inputBuffer.destroy();
  uniformBuffer.destroy();
  outputBuffer.destroy();

  console.log(
    `[RefineCompute] Processed ${pixelCount} pixels ` +
      `(${(byteSize / 1024 / 1024).toFixed(1)} MB).`,
  );

  return result;
}
