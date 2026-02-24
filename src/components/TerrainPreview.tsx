import { useMemo, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import {
  terrainVertexShader,
  terrainFragmentShader,
} from "./terrainShaders";

/* ------------------------------------------------------------------ */
/*  TerrainMesh – the actual displaced plane inside the R3F scene     */
/* ------------------------------------------------------------------ */

interface TerrainMeshProps {
  heightData: Float32Array;
  /** Side length of the square heightmap (e.g. 4096). */
  mapSize: number;
}

/**
 * Resolution of the preview plane.  We do NOT use a 4096×4096 mesh
 * (that's 16 M vertices) — instead we down‑sample into a manageable
 * grid and let the GPU texture sampler interpolate the rest.
 */
const PREVIEW_SEGMENTS = 512;

function TerrainMesh({ heightData, mapSize }: TerrainMeshProps) {
  const matRef = useRef<THREE.ShaderMaterial>(null);

  /* ---- build a DataTexture from the Float32 heightmap ---- */
  const heightTexture = useMemo(() => {
    // Convert the 1‑channel float array → RGBA Uint8 for DataTexture.
    const pixels = new Uint8Array(mapSize * mapSize * 4);
    for (let i = 0; i < heightData.length; i++) {
      const v = Math.round(heightData[i] * 255);
      const j = i * 4;
      pixels[j] = v;
      pixels[j + 1] = v;
      pixels[j + 2] = v;
      pixels[j + 3] = 255;
    }

    const tex = new THREE.DataTexture(
      pixels,
      mapSize,
      mapSize,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
    );
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    return tex;
  }, [heightData, mapSize]);

  /* ---- ShaderMaterial uniforms ---- */
  const uniforms = useMemo(
    () => ({
      uHeightmap: { value: heightTexture },
      uHeightScale: { value: 1.5 }, // visual exaggeration
    }),
    [heightTexture],
  );

  /* ---- update texture when data changes ---- */
  useEffect(() => {
    if (matRef.current) {
      matRef.current.uniforms.uHeightmap.value = heightTexture;
      matRef.current.needsUpdate = true;
    }
  }, [heightTexture]);

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[4, 4, PREVIEW_SEGMENTS, PREVIEW_SEGMENTS]} />
      <shaderMaterial
        ref={matRef}
        vertexShader={terrainVertexShader}
        fragmentShader={terrainFragmentShader}
        uniforms={uniforms}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/* ------------------------------------------------------------------ */
/*  TerrainPreview – the React Three Fiber <Canvas> wrapper           */
/* ------------------------------------------------------------------ */

export interface TerrainPreviewProps {
  heightData: Float32Array;
  mapSize: number;
}

export function TerrainPreview({ heightData, mapSize }: TerrainPreviewProps) {
  return (
    <div className="terrain-preview">
      <Canvas
        camera={{ position: [0, 2.5, 3.5], fov: 50, near: 0.01, far: 100 }}
        gl={{ antialias: true }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 8, 3]} intensity={0.8} />
        <TerrainMesh heightData={heightData} mapSize={mapSize} />
        <OrbitControls
          enableDamping
          dampingFactor={0.12}
          minDistance={1}
          maxDistance={12}
        />
      </Canvas>
    </div>
  );
}
