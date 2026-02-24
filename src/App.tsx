import { useMemo, useState, useCallback } from "react";
import { Canvas2DEngine } from "./engine/Canvas2DEngine";
import type { BrushSettings } from "./engine/types";
import { DrawingCanvas } from "./components/DrawingCanvas";
import { Toolbar } from "./components/Toolbar";
import { TerrainPreview } from "./components/TerrainPreview";
import { runHeightmapCompute, readbackOutputBuffer } from "./gpu";
import { downloadHeightmap16 } from "./gpu/exportHeightmap";
import { runRefineCompute } from "./gpu/refineCompute";

interface TerrainData {
  heightData: Float32Array;
  mapSize: number;
}

function App() {
  const engine = useMemo(() => new Canvas2DEngine(), []);

  const [brushSize, setBrushSize] = useState(64);
  const [brushSoftness, setBrushSoftness] = useState(0.5);
  const [brushColor, setBrushColor] = useState(255);

  const brush: BrushSettings = useMemo(
    () => ({ size: brushSize, softness: brushSoftness, color: brushColor }),
    [brushSize, brushSoftness, brushColor],
  );

  const [isGenerating, setIsGenerating] = useState(false);
  const [isRefining, setIsRefining] = useState(false);
  const [terrain, setTerrain] = useState<TerrainData | null>(null);

  const handleClear = useCallback(() => engine.clear(), [engine]);

  const handleRefine = useCallback(async () => {
    setIsRefining(true);
    try {
      const hiddenCanvas = engine.getHiddenCanvas();
      const refined = await runRefineCompute(hiddenCanvas);
      engine.writeBack(refined);
      console.log("[Refine] Canvas updated with refined terrain.");
    } catch (err) {
      console.error("[Refine] Failed:", err);
      alert(err instanceof Error ? err.message : "WebGPU refine error");
    } finally {
      setIsRefining(false);
    }
  }, [engine]);

  const handleDownload = useCallback(() => {
    if (!terrain) return;
    downloadHeightmap16(terrain.heightData, terrain.mapSize, terrain.mapSize);
  }, [terrain]);

  const handleGenerate3D = useCallback(async () => {
    setIsGenerating(true);
    try {
      const hiddenCanvas = engine.getHiddenCanvas();
      const result = await runHeightmapCompute(hiddenCanvas);

      // Readback compute output to CPU so Three.js (WebGL) can use it.
      const heightData = await readbackOutputBuffer(
        result.device,
        result.outputBuffer,
        result.pixelCount,
      );

      const mapSize = hiddenCanvas.width; // 4096
      setTerrain({ heightData, mapSize });

      // Free the GPU output buffer now that we have a CPU copy.
      result.outputBuffer.destroy();

      console.log(
        "[Generate3D] Terrain ready.",
        `${result.pixelCount} pixels read back.`,
      );
    } catch (err) {
      console.error("[Generate3D] Failed:", err);
      alert(err instanceof Error ? err.message : "WebGPU error");
    } finally {
      setIsGenerating(false);
    }
  }, [engine]);

  return (
    <div className="app-layout">
      <Toolbar
        brushSize={brushSize}
        setBrushSize={setBrushSize}
        brushSoftness={brushSoftness}
        setBrushSoftness={setBrushSoftness}
        brushColor={brushColor}
        setBrushColor={setBrushColor}
        onClear={handleClear}
        onRefine={handleRefine}
        isRefining={isRefining}
        onGenerate3D={handleGenerate3D}
        isGenerating={isGenerating}
        onDownload={handleDownload}
        hasTerrainData={terrain !== null}
      />
      <main className="canvas-container">
        <DrawingCanvas engine={engine} brush={brush} />
      </main>

      {terrain && (
        <TerrainPreview
          heightData={terrain.heightData}
          mapSize={terrain.mapSize}
        />
      )}
    </div>
  );
}

export default App;
