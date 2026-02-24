/**
 * GLSL vertex shader for terrain displacement.
 *
 * Reads a grayscale heightmap texture and displaces vertices along Y.
 * Also forwards the height value to the fragment shader for colouring.
 */
export const terrainVertexShader = /* glsl */ `
  uniform sampler2D uHeightmap;
  uniform float     uHeightScale;

  varying float vHeight;
  varying vec2  vUv;

  void main() {
    vUv = uv;

    // Sample the heightmap (R channel, all channels are identical).
    float h = texture2D(uHeightmap, uv).r;
    vHeight = h;

    // Displace along Y (up).
    vec3 displaced = position;
    displaced.y += h * uHeightScale;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
  }
`;

/**
 * GLSL fragment shader for terrain colouring.
 *
 * Simple gradient: dark green (low) → brown (mid) → white (peaks).
 */
export const terrainFragmentShader = /* glsl */ `
  varying float vHeight;
  varying vec2  vUv;

  void main() {
    // Colour ramp ---------------------------------------------------
    vec3 low  = vec3(0.12, 0.18, 0.08);   // dark green  (valley)
    vec3 mid  = vec3(0.45, 0.36, 0.22);   // brown       (slopes)
    vec3 high = vec3(0.95, 0.95, 0.97);   // near‑white  (peaks)

    vec3 colour;
    if (vHeight < 0.5) {
      colour = mix(low, mid, vHeight * 2.0);
    } else {
      colour = mix(mid, high, (vHeight - 0.5) * 2.0);
    }

    // Cheap pseudo‑ambient term so it doesn't look completely flat.
    float ambient = 0.35 + 0.65 * vHeight;
    colour *= ambient;

    gl_FragColor = vec4(colour, 1.0);
  }
`;
