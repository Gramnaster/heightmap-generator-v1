/**
 * Acquire a WebGPU device + adapter, or throw a descriptive error.
 *
 * Call once at app start (or lazily on first use) and cache the result.
 */
export async function initWebGPU(): Promise<{
  adapter: GPUAdapter;
  device: GPUDevice;
}> {
  if (!navigator.gpu) {
    throw new Error(
      "WebGPU is not supported in this browser. " +
        "Try Chrome 113+ or Edge 113+ with hardware acceleration enabled.",
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });

  if (!adapter) {
    throw new Error(
      "Could not obtain a GPUAdapter. " +
        "Make sure your GPU drivers are up to date.",
    );
  }

  const device = await adapter.requestDevice();

  device.lost.then((info) => {
    console.error(`WebGPU device lost: ${info.message}`);
    if (info.reason !== "destroyed") {
      // Attempt recovery in a future iteration
      console.warn("Consider re‑initialising the device.");
    }
  });

  return { adapter, device };
}
