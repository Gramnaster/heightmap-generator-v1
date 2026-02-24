import { useMemo, useState, useCallback } from "react";
import { Canvas2DEngine } from "./engine/Canvas2DEngine";
import type { BrushSettings } from "./engine/types";
import { DrawingCanvas } from "./components/DrawingCanvas";
import { Toolbar } from "./components/Toolbar";
import { runHeightmapCompute } from "./gpu";

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

  const handleClear = useCallback(() => engine.clear(), [engine]);

  const handleGenerate3D = useCallback(async () => {
    setIsGenerating(true);
    try {
      const hiddenCanvas = engine.getHiddenCanvas();
      const result = await runHeightmapCompute(hiddenCanvas);
      console.log(
        "[Generate3D] Compute complete.",
        `Output buffer: ${result.outputBuffer.size} bytes,`,
        `${result.pixelCount} pixels on GPU.`,
      );
      // TODO: feed result.outputBuffer into a 3D renderer
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
        onGenerate3D={handleGenerate3D}
        isGenerating={isGenerating}
      />
      <main className="canvas-container">
        <DrawingCanvas engine={engine} brush={brush} />
      </main>
    </div>
  );
}

export default App;
