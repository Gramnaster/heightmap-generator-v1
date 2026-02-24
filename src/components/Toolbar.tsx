interface ToolbarProps {
  brushSize: number;
  setBrushSize: (v: number) => void;
  brushSoftness: number;
  setBrushSoftness: (v: number) => void;
  brushColor: number;
  setBrushColor: (v: number) => void;
  onClear: () => void;
  onRefine: () => void;
  isRefining: boolean;
  onGenerate3D: () => void;
  isGenerating: boolean;
  onDownload: () => void;
  hasTerrainData: boolean;
}

/**
 * Sidebar toolbar with brush controls.
 *
 * All colours are strictly grayscale (0–255).
 */
export function Toolbar({
  brushSize,
  setBrushSize,
  brushSoftness,
  setBrushSoftness,
  brushColor,
  setBrushColor,
  onClear,
  onRefine,
  isRefining,
  onGenerate3D,
  isGenerating,
  onDownload,
  hasTerrainData,
}: ToolbarProps) {
  const grayHex = brushColor.toString(16).padStart(2, "0");
  const swatchColor = `#${grayHex}${grayHex}${grayHex}`;

  return (
    <aside className="toolbar">
      <h2 className="toolbar-title">Brush</h2>

      {/* ---- Size ---- */}
      <label className="toolbar-label">
        Size
        <span className="toolbar-value">{brushSize} px</span>
      </label>
      <input
        type="range"
        min={1}
        max={512}
        value={brushSize}
        onChange={(e) => setBrushSize(Number(e.target.value))}
      />

      {/* ---- Softness / Feathering ---- */}
      <label className="toolbar-label">
        Softness
        <span className="toolbar-value">{Math.round(brushSoftness * 100)}%</span>
      </label>
      <input
        type="range"
        min={0}
        max={100}
        value={Math.round(brushSoftness * 100)}
        onChange={(e) => setBrushSoftness(Number(e.target.value) / 100)}
      />

      {/* ---- Color (grayscale) ---- */}
      <label className="toolbar-label">
        Color
        <span className="toolbar-value">{brushColor}</span>
      </label>
      <div className="toolbar-color-row">
        <input
          type="range"
          min={0}
          max={255}
          value={brushColor}
          onChange={(e) => setBrushColor(Number(e.target.value))}
        />
        <div
          className="toolbar-swatch"
          style={{ backgroundColor: swatchColor }}
        />
      </div>

      {/* ---- Actions ---- */}
      <hr className="toolbar-divider" />
      <button className="toolbar-btn" onClick={onClear}>
        Clear Canvas
      </button>
      <button
        className="toolbar-btn toolbar-btn--secondary"
        onClick={onRefine}
        disabled={isRefining}
      >
        {isRefining ? "Refining…" : "Refine Terrain"}
      </button>
      <button
        className="toolbar-btn toolbar-btn--primary"
        onClick={onGenerate3D}
        disabled={isGenerating}
      >
        {isGenerating ? "Generating…" : "Generate 3D"}
      </button>
      <button
        className="toolbar-btn"
        onClick={onDownload}
        disabled={!hasTerrainData}
      >
        Download 16‑bit PNG
      </button>
    </aside>
  );
}
