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
 *   3.  Coastline contrast      – sharpens ONLY the land/sea border
 *                                  (the near‑zero boundary) so the
 *                                  coast snaps to black or the terrain
 *                                  value.  Interior terrain (all gray
 *                                  levels) is fully preserved.
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
/* ================================================================== */

fn gaussianBlur(px : i32, py : i32) -> f32 {
  var total  = 0.0;
  var weight = 0.0;

  for (var dy = -3; dy <= 3; dy = dy + 1) {
    for (var dx = -3; dx <= 3; dx = dx + 1) {
      let d2 = f32(dx * dx + dy * dy);
      let w  = exp(-d2 / 8.0);
      total  = total + sampleClamped(px + dx, py + dy) * w;
      weight = weight + w;
    }
  }
  return total / weight;
}

/* ================================================================== */
/*  Edge detection + noise perturbation                               */
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

// Multi‑octave FBM displacement — produces fractal, jagged coastlines
// rather than smooth single‑frequency wobbles.
fn fbmDisplace(uv : vec2f, seed : vec2f) -> f32 {
  var value = 0.0;
  var amp   = 1.0;
  var freq  = 1.0;
  var p     = uv + seed;
  // 5 octaves: big bays down to tiny inlets
  for (var i = 0; i < 5; i = i + 1) {
    value = value + amp * (valueNoise(p * freq) * 2.0 - 1.0);
    freq  = freq * 2.0;
    amp   = amp  * 0.5;
    // rotate per octave to avoid grid alignment
    p = vec2f(p.x * 0.866 - p.y * 0.5,
              p.x * 0.5   + p.y * 0.866);
  }
  return value;  // range roughly ‑2..+2
}

fn perturbedSample(px : i32, py : i32) -> f32 {
  let uv = vec2f(f32(px), f32(py)) / vec2f(f32(params.width), f32(params.height));
  let edge = gradientMagnitude(px, py);

  // Fractal displacement: base frequency 8, max shift 24 px at
  // hard edges — creates realistic jagged coastlines with bays,
  // peninsulas and inlets at multiple scales.
  let baseFreq = 8.0;
  let maxShift = 24.0;
  let nx = fbmDisplace(uv * baseFreq, vec2f(0.0, 0.0))   * maxShift * edge;
  let ny = fbmDisplace(uv * baseFreq, vec2f(73.1, 19.7))  * maxShift * edge;

  return sampleClamped(px + i32(round(nx)), py + i32(round(ny)));
}

/* ================================================================== */
/*  Coastline contrast                                                */
/*                                                                    */
/*  Only sharpens the land / sea boundary.  Pixels that are clearly   */
/*  "terrain" (above a low threshold) pass through with their full    */
/*  value.  Pixels in the narrow transition band near zero get pushed */
/*  toward either 0 (sea) or their terrain value (land).              */
/*  This preserves ALL gray terrain levels.                           */
/* ================================================================== */

fn coastlineContrast(v : f32) -> f32 {
  // Threshold: anything below this is "sea", above is "land".
  // The smoothstep creates a sharp but smooth step right at the
  // coastline without affecting terrain values above the band.
  let coastLow  = 0.03;  // below this → hard black
  let coastHigh = 0.12;  // above this → fully preserved terrain
  return v * smoothstep(coastLow, coastHigh, v);
}

/* ================================================================== */
/*  Mountain sharpening (unsharp‑mask)                                */
/* ================================================================== */

fn sharpenMountains(original : f32, blurred : f32) -> f32 {
  let detail   = original - blurred;
  // Aggressive sharpening on bright terrain — kicks in earlier
  // (0.20) and reaches full strength (1.5×) at mountain peaks.
  let strength = smoothstep(0.20, 0.55, original) * 1.5;
  let sharpened = original + detail * strength;
  // Also push mountain contrast: expand the bright range so peaks
  // separate more clearly from foothills.
  let mtnBoost = smoothstep(0.40, 0.80, sharpened);
  let boosted  = mix(sharpened, pow(sharpened, 0.6), mtnBoost * 0.4);
  return boosted;
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
  // Smooths jagged brush edges into cleaner gradients.
  let blurred = gaussianBlur(px, py);

  // ── Step 2: Noise‑perturbed edges ─────────────────────────────
  // At edge pixels, replace the blurred value with a noise‑shifted
  // sample so coastlines become organic instead of perfectly round.
  let edge      = gradientMagnitude(px, py);
  let perturbed = perturbedSample(px, py);
  let edgeMix   = smoothstep(0.02, 0.15, edge);
  let smoothed  = mix(blurred, perturbed, edgeMix * 0.7);

  // ── Step 3: Mountain sharpening ───────────────────────────────
  // Boosts detail on bright (mountain) pixels only.
  let sharpened = sharpenMountains(smoothed, blurred);

  // ── Step 4: Coastline contrast ────────────────────────────────
  // Sharpens ONLY the land/sea boundary (near‑zero transition).
  // All interior terrain (dark gray, mid gray, white) is preserved.
  let refined = coastlineContrast(sharpened);

  heightmapOut[idx] = clamp(refined, 0.0, 1.0);
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
