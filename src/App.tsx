import { useMemo, useState, useCallback } from "react";
import { Canvas2DEngine } from "./engine/Canvas2DEngine";
import type { BrushSettings } from "./engine/types";
import { DrawingCanvas } from "./components/DrawingCanvas";
import { Toolbar } from "./components/Toolbar";

function App() {
  const engine = useMemo(() => new Canvas2DEngine(), []);

  const [brushSize, setBrushSize] = useState(64);
  const [brushSoftness, setBrushSoftness] = useState(0.5);
  const [brushColor, setBrushColor] = useState(255);

  const brush: BrushSettings = useMemo(
    () => ({ size: brushSize, softness: brushSoftness, color: brushColor }),
    [brushSize, brushSoftness, brushColor],
  );

  const handleClear = useCallback(() => engine.clear(), [engine]);

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
      />
      <main className="canvas-container">
        <DrawingCanvas engine={engine} brush={brush} />
      </main>
    </div>
  );
}

export default App;
