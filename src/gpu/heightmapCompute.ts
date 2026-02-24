import { initWebGPU } from "./initWebGPU";
import { extractHeightmap } from "./extractHeightmap";

/* ------------------------------------------------------------------ */
/*  Compute shader (WGSL)                                             */
/* ------------------------------------------------------------------ */

/**
 * Minimal pass‑through compute shader.
 *
 * It reads each heightmap float from the input storage buffer, and
 * writes it unmodified into the output storage buffer.  Swap in real
 * terrain‑generation logic later — the pipeline plumbing stays the same.
 *
 * Bindings:
 *   @group(0) @binding(0)  heightmapIn   – read‑only  storage<f32[]>
 *   @group(0) @binding(1)  heightmapOut  – read‑write storage<f32[]>
 */
const COMPUTE_SHADER_SRC = /* wgsl */ `

@group(0) @binding(0) var<storage, read>       heightmapIn  : array<f32>;
@group(0) @binding(1) var<storage, read_write> heightmapOut : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&heightmapIn)) {
    return;
  }
  // --- placeholder: identity copy ---
  // Replace this with actual terrain processing (e.g. erosion,
  // normal‑map generation, mesh displacement, etc.).
  heightmapOut[idx] = heightmapIn[idx];
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
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
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

  /* ---- 3.  Create GPU buffers ---- */
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
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
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

  /* ---- 5.  Clean up the input buffer (output stays alive) ---- */
  inputBuffer.destroy();

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
