/**
 * Anduril Lattice Dark-Obsidian 3D KSFO Digital Twin (`OpenXVS Studio v4.0`).
 *
 * World Coordinates: Three.js Y-up meters centered on KSFO Runway 28L threshold (0, 0, 0):
 *   +X = East (m), +Y = Up above 28L runway datum (m), -Z = North (m).
 *
 * Includes:
 *   - 256x256 real USGS 3DEP inner terrain + 128x128 outer SF Bay mountains
 *   - Tactical Obsidian + Glowing Cyan/Ice Topographic Contour Shader (`default`)
 *     + smooth `Satellite Hybrid` toggle blending real 2048x2048 0.98m/texel USGS NAIP ortho
 *   - Real 3D KSFO Airport (Runways 28L/28R/19L/19R, taxiways, blooming ALSF-2 running strobes,
 *     4-light PAPI, iconic sculptural KSFO ATC Tower w/ rotating radar, 3D Terminal Concourses)
 *   - 3D San Francisco Downtown Skyline + Bay Bridge suspension towers
 *   - Glossy Reflective Dark-Navy SF Bay Water (zero moiré)
 *   - Deep Twilight / Blue-Hour Atmospheric Sky Dome w/ warm western sunset band & stars
 */

import * as THREE from "three";
import {
  KSFO_28L_ORIGIN,
  latLonToKsfoLocalEn,
  sampleKsfoElevationM,
  type KsfoTerrainPack,
} from "../sim/ksfo_terrain_pack.ts";
import type { AtmosphereLightingState, VisualWorldMode } from "./atmosphere_world.ts";

export interface KsfoWorldRig {
  readonly group: THREE.Group;
  readonly terrainMaterial: THREE.ShaderMaterial;
  setTerrainPack(pack: KsfoTerrainPack): void;
  setOrthoTextures(core2k: THREE.Texture | null, inner10k: THREE.Texture | null): void;
  setVisualMode(mode: VisualWorldMode): void;
  getVisualMode(): VisualWorldMode;
  update(simTimeS: number, lighting: AtmosphereLightingState, fogDensity: number): void;
}

const DEG_TO_RAD = Math.PI / 180.0;
const RWY_28_HDG_RAD = KSFO_28L_ORIGIN.trueHeadingDeg * DEG_TO_RAD; // 297.7°
const RWY_19_HDG_RAD = 207.7 * DEG_TO_RAD; // perpendicular crossing runways 19L/19R

// Exact KSFO Core 2km NAIP Ortho center (matches ksfo_ortho_core2k_v2.jpg)
const [CORE2K_CX, CORE2K_CN] = latLonToKsfoLocalEn(37.6155, -122.3625);
const CORE2K_CZ = -CORE2K_CN;
const [INNER10K_CX, INNER10K_CN] = latLonToKsfoLocalEn(37.6188056, -122.3754167);
const INNER10K_CZ = -INNER10K_CN;

/** Procedural high-contrast asphalt runway texture (piano keys, centerline, numbers, touchdown bars). */
function createRunwayCanvasTexture(designatorA: string, designatorB: string): THREE.CanvasTexture | null {
  if (typeof document === "undefined") return null;
  const w = 1024;
  const h = 256;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  // Dark weathered tactical asphalt base
  ctx.fillStyle = "#111827";
  ctx.fillRect(0, 0, w, h);

  // Subtle tire rubber skid zone along centerline
  const skid = ctx.createLinearGradient(0, 0, 0, h);
  skid.addColorStop(0, "rgba(255,255,255,0.03)");
  skid.addColorStop(0.32, "rgba(0,0,0,0.22)");
  skid.addColorStop(0.5, "rgba(0,0,0,0.42)");
  skid.addColorStop(0.68, "rgba(0,0,0,0.22)");
  skid.addColorStop(1, "rgba(255,255,255,0.03)");
  ctx.fillStyle = skid;
  ctx.fillRect(0, 0, w, h);

  // Authentic heavy rubber touchdown skid marks inside touchdown zones of 28L/28R
  for (const xTdZone of [220, w - 360]) {
    const rubber = ctx.createLinearGradient(xTdZone, 0, xTdZone + 140, 0);
    rubber.addColorStop(0, "rgba(4, 6, 10, 0.0)");
    rubber.addColorStop(0.35, "rgba(4, 6, 10, 0.52)");
    rubber.addColorStop(1, "rgba(4, 6, 10, 0.0)");
    ctx.fillStyle = rubber;
    ctx.fillRect(xTdZone, h * 0.28, 140, h * 0.44);
  }

  // Crisp side edge stripes
  ctx.fillStyle = "#dbeafe";
  ctx.fillRect(0, 12, w, 5);
  ctx.fillRect(0, h - 17, w, 5);

  // Piano-key threshold bars at both ends
  for (const xStart of [18, w - 68]) {
    for (let i = 0; i < 10; i++) {
      const y = 26 + i * 21;
      ctx.fillRect(xStart, y, 48, 12);
    }
  }

  // Aiming point blocks & touchdown zone bars
  for (const xAim of [172, w - 232]) {
    ctx.fillRect(xAim, 54, 60, 22);
    ctx.fillRect(xAim, h - 76, 60, 22);
  }
  for (const xTd of [254, 306, w - 284, w - 336]) {
    ctx.fillRect(xTd, 64, 28, 10);
    ctx.fillRect(xTd, h - 74, 28, 10);
  }

  // Dashed centerline
  for (let x = 148; x < w - 148; x += 48) {
    ctx.fillRect(x, h / 2 - 3, 28, 6);
  }

  // Bold white runway designator numbers
  ctx.fillStyle = "#f1f5f9";
  ctx.font = "900 44px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  ctx.save();
  ctx.translate(108, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(designatorA, 0, 0);
  ctx.restore();

  ctx.save();
  ctx.translate(w - 108, h / 2);
  ctx.rotate(Math.PI / 2);
  ctx.fillText(designatorB, 0, 0);
  ctx.restore();

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

/** Builds a displaced Three.js BufferGeometry plane covering `(sizeM x sizeM)` centered at `(cx, cz)`. */
function buildTerrainGeometry(
  sizeM: number,
  segments: number,
  centerX: number,
  centerZ: number,
  pack: KsfoTerrainPack | null,
): THREE.PlaneGeometry {
  const geo = new THREE.PlaneGeometry(sizeM, sizeM, segments, segments);
  geo.rotateX(-Math.PI * 0.5);
  displaceTerrainGeometry(geo, centerX, centerZ, pack);
  return geo;
}

function displaceTerrainGeometry(
  geo: THREE.PlaneGeometry,
  centerX: number,
  centerZ: number,
  pack: KsfoTerrainPack | null,
): void {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i) + centerX;
    const z = pos.getZ(i) + centerZ;
    const elevM = sampleKsfoElevationM(pack, x, -z);
    // Keep airport tarmac flat & crisp right around KSFO runways while preserving full San Bruno hills!
    const distAirport = Math.hypot(x + 1150.0, z + 550.0);
    let y = elevM;
    if (distAirport < 1900.0 && y > -2.5 && y < 12.0) {
      const flatBlend = Math.max(0.0, Math.min(1.0, (distAirport - 1350.0) / 550.0));
      y = y * flatBlend + -0.35 * (1.0 - flatBlend);
    }
    pos.setY(i, y);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
}

/** Creates the dual-mode terrain shader (Photorealistic California Coast + USGS NAIP vs Tactical Obsidian). */
function createObsidianTerrainMaterial(): THREE.ShaderMaterial {
  const dummyTex = new THREE.DataTexture(new Uint8Array([12, 18, 30, 255]), 1, 1);
  dummyTex.needsUpdate = true;

  return new THREE.ShaderMaterial({
    uniforms: {
      uSunDir: { value: new THREE.Vector3(-0.78, 0.32, -0.52).normalize() },
      uSunColor: { value: new THREE.Color(0xfff1dc) },
      uDaylight: { value: 1.0 },
      uPhotorealMix: { value: 1.0 },
      uHasCoreOrtho: { value: 0.0 },
      uHasInnerOrtho: { value: 0.0 },
      uCoreOrtho: { value: dummyTex },
      uInnerOrtho: { value: dummyTex },
      uCoreCenterXZ: { value: new THREE.Vector2(CORE2K_CX, CORE2K_CZ) },
      uInnerCenterXZ: { value: new THREE.Vector2(INNER10K_CX, INNER10K_CZ) },
      uFogColor: { value: new THREE.Color(0x9cb8d4) },
      uFogDensity: { value: 0.000052 },
    },
    vertexShader: `
      varying vec3 vWorldPos;
      varying vec3 vNormalW;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        vNormalW = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: `
      uniform vec3 uSunDir;
      uniform vec3 uSunColor;
      uniform float uDaylight;
      uniform float uPhotorealMix;
      uniform float uHasCoreOrtho;
      uniform float uHasInnerOrtho;
      uniform sampler2D uCoreOrtho;
      uniform sampler2D uInnerOrtho;
      uniform vec2 uCoreCenterXZ;
      uniform vec2 uInnerCenterXZ;
      uniform vec3 uFogColor;
      uniform float uFogDensity;

      varying vec3 vWorldPos;
      varying vec3 vNormalW;

      float contourBand(float h, float stepM, float widthPx) {
        float scaled = h / stepM;
        float fracDist = abs(fract(scaled - 0.5) - 0.5);
        float fw = max(fwidth(scaled), 1e-4);
        return 1.0 - smoothstep(0.0, fw * widthPx, fracDist);
      }

      float gridBand(vec2 xz, float stepM) {
        vec2 g = xz / stepM;
        vec2 d = abs(fract(g - 0.5) - 0.5) / max(fwidth(g), vec2(1e-4));
        float line = min(d.x, d.y);
        return 1.0 - smoothstep(0.0, 1.35, line);
      }

      void main() {
        vec3 N = normalize(vNormalW);
        float h = vWorldPos.y;
        float distCam = length(cameraPosition - vWorldPos);

        // Smooth shoreline feathering eliminates jagged elevation steps in BOTH Photoreal & Obsidian!
        float shoreFeather = smoothstep(-0.6, 3.0, h);
        if (h < -3.2) {
          discard;
        }

        float elevT = clamp((h + 1.0) / 360.0, 0.0, 1.0);
        float ndl = clamp(dot(N, normalize(uSunDir)), 0.0, 1.0);
        float slope = 1.0 - clamp(N.y, 0.0, 1.0);

        // 1. Tactical Obsidian Mode
        vec3 baseLow = vec3(0.032, 0.055, 0.098);
        vec3 baseMid = vec3(0.058, 0.102, 0.172);
        vec3 baseHigh = vec3(0.105, 0.175, 0.275);
        vec3 obsidian = mix(baseLow, baseMid, smoothstep(0.0, 0.35, elevT));
        obsidian = mix(obsidian, baseHigh, smoothstep(0.35, 1.0, elevT));
        vec3 litObsidian = obsidian * (0.68 + 0.62 * ndl) + vec3(0.02, 0.08, 0.15) * slope;

        float landMask = smoothstep(0.4, 4.2, h);
        float minorC = contourBand(h, 25.0, 1.35) * landMask;
        float majorC = contourBand(h, 100.0, 1.85) * landMask;
        float gridFade = exp(-distCam * 0.00014);
        float tacticalGrid = gridBand(vWorldPos.xz, 500.0) * gridFade * 0.20 * shoreFeather;
        vec3 contourCol = vec3(0.08, 0.42, 0.68) * minorC * 0.42
                        + vec3(0.22, 0.78, 0.98) * majorC * 0.85
                        + vec3(0.14, 0.56, 0.82) * tacticalGrid;
        vec3 obsidianFinal = litObsidian + contourCol;

        // 2. Photorealistic California Coastal Peninsula + Feathered Real USGS NAIP Ortho (zero rectangular slab!)
        float microNoise = sin(vWorldPos.x * 0.0045) * cos(vWorldPos.z * 0.0045) * 0.5 + 0.5;
        float fineNoise = sin(vWorldPos.x * 0.022 + vWorldPos.z * 0.017) * 0.5 + 0.5;
        vec3 coastGrass = vec3(0.22, 0.31, 0.19);
        vec3 sageHill = vec3(0.28, 0.36, 0.21);
        vec3 goldenRidge = vec3(0.46, 0.42, 0.28);
        vec3 coastalAlbedo = mix(coastGrass, sageHill, smoothstep(0.02, 0.32, elevT));
        coastalAlbedo = mix(coastalAlbedo, goldenRidge, smoothstep(0.32, 0.88, elevT) * (0.65 + 0.35 * microNoise));
        coastalAlbedo = mix(coastalAlbedo, vec3(0.32, 0.30, 0.27), smoothstep(0.22, 0.65, slope));
        coastalAlbedo *= 0.88 + 0.22 * fineNoise;

        // Smooth radial + border feathering of verified local USGS NAIP ortho (eliminates hard rectangular box!)
        vec2 uvInner = (vWorldPos.xz - uInnerCenterXZ) / 10000.0 + 0.5;
        uvInner.y = 1.0 - uvInner.y;
        if (uHasInnerOrtho > 0.5 && all(greaterThan(uvInner, vec2(0.01))) && all(lessThan(uvInner, vec2(0.99)))) {
          vec3 sInner = texture2D(uInnerOrtho, uvInner).rgb;
          vec2 eI = min(uvInner, 1.0 - uvInner);
          float boxFeather = smoothstep(0.02, 0.16, min(eI.x, eI.y));
          float radInner = 1.0 - smoothstep(0.34, 0.49, length(uvInner - vec2(0.5)));
          float landOnly = smoothstep(0.2, 3.8, h);
          coastalAlbedo = mix(coastalAlbedo, sInner * vec3(0.94, 0.96, 0.90), boxFeather * radInner * landOnly * 0.84);
        }
        vec2 uvCore = (vWorldPos.xz - uCoreCenterXZ) / 2000.0 + 0.5;
        uvCore.y = 1.0 - uvCore.y;
        if (uHasCoreOrtho > 0.5 && all(greaterThan(uvCore, vec2(0.01))) && all(lessThan(uvCore, vec2(0.99)))) {
          vec3 sCore = texture2D(uCoreOrtho, uvCore).rgb;
          vec2 eC = min(uvCore, 1.0 - uvCore);
          float boxC = smoothstep(0.02, 0.16, min(eC.x, eC.y));
          float radC = 1.0 - smoothstep(0.34, 0.49, length(uvCore - vec2(0.5)));
          float landC = smoothstep(-0.3, 2.2, h);
          coastalAlbedo = mix(coastalAlbedo, sCore * vec3(0.95, 0.97, 0.94), boxC * radC * landC * 0.92);
        }

        // Physical California daylight / golden-hour directional + sky hemisphere illumination
        vec3 ambientDay = mix(vec3(0.05, 0.08, 0.14), vec3(0.26, 0.34, 0.44), uDaylight);
        vec3 sunLit = uSunColor * (0.26 + 0.92 * ndl) * (0.18 + 0.82 * uDaylight);
        vec3 photorealFinal = coastalAlbedo * (ambientDay + sunLit);

        // Smooth shoreline transition matching deep Pacific cobalt-teal bay water
        vec3 bayShoreObs = vec3(0.022, 0.062, 0.118);
        vec3 bayShoreDay = mix(vec3(0.02, 0.06, 0.10), vec3(0.045, 0.185, 0.275), uDaylight);
        vec3 bayShore = mix(bayShoreObs, bayShoreDay, uPhotorealMix);
        vec3 landCol = mix(obsidianFinal, photorealFinal, uPhotorealMix);
        vec3 col = mix(bayShore, landCol, shoreFeather);

        // Soft coastal surf highlight along waterline
        float shoreRim = exp(-pow((h - 0.5) * 1.2, 2.0));
        vec3 foamCol = mix(vec3(0.12, 0.52, 0.78) * 0.18, vec3(0.76, 0.84, 0.88) * (0.06 + 0.12 * uDaylight), uPhotorealMix);
        col += foamCol * shoreRim;

        float fogFactor = 1.0 - exp(-pow(distCam * uFogDensity, 1.38));
        col = mix(col, uFogColor, clamp(fogFactor, 0.0, 0.96));
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
}

/** Realistic animated San Francisco Bay water shader (sparkling daylight Fresnel + dark-navy obsidian). */
function createSfBayWaterMesh(): { readonly mesh: THREE.Mesh; readonly mat: THREE.ShaderMaterial } {
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0.0 },
      uSunDir: { value: new THREE.Vector3(-0.78, 0.32, -0.52).normalize() },
      uSunColor: { value: new THREE.Color(0xfff1dc) },
      uDaylight: { value: 1.0 },
      uPhotorealMix: { value: 1.0 },
      uFogColor: { value: new THREE.Color(0x9cb8d4) },
      uFogDensity: { value: 0.000052 },
    },
    vertexShader: `
      varying vec3 vWorldPos;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: `
      uniform float uTime;
      uniform vec3 uSunDir;
      uniform vec3 uSunColor;
      uniform float uDaylight;
      uniform float uPhotorealMix;
      uniform vec3 uFogColor;
      uniform float uFogDensity;
      varying vec3 vWorldPos;

      void main() {
        vec3 V = normalize(cameraPosition - vWorldPos);
        float dist = length(cameraPosition - vWorldPos);
        // Dual-octave scrolling wave normals with smooth distance damping (zero moiré)
        float rippleFade = exp(-dist * 0.00024);
        vec2 p = vWorldPos.xz;
        float w1 = sin(p.x * 0.036 + p.y * 0.024 + uTime * 1.25) * 0.45;
        float w2 = cos(p.x * 0.082 - p.y * 0.064 - uTime * 1.85) * 0.32;
        float w3 = sin((p.x * 0.14 + p.y * 0.11) + uTime * 2.4) * 0.18;
        float w4 = cos((p.x * 0.22 - p.y * 0.19) - uTime * 2.9) * 0.12;
        vec3 N = normalize(vec3((w1 + w2 + w4) * 0.045 * rippleFade, 1.0, (w2 - w3 + w4) * 0.045 * rippleFade));

        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.8);

        // Obsidian deep navy vs rich deep Pacific cobalt-teal (#0b3656 -> #1a5c80 -> #3583a0 coastal sheen)
        vec3 obsDeep = vec3(0.014, 0.036, 0.072);
        vec3 obsRefl = vec3(0.07, 0.19, 0.34);

        vec3 dayDeep = mix(vec3(0.010, 0.035, 0.070), vec3(0.008, 0.052, 0.118), uDaylight);
        vec3 dayMid = mix(vec3(0.020, 0.065, 0.115), vec3(0.024, 0.128, 0.225), uDaylight);
        vec3 dayRefl = mix(vec3(0.06, 0.16, 0.28), vec3(0.18, 0.42, 0.58), uDaylight);

        float midMask = clamp(0.5 + 0.5 * sin(p.x * 0.00045 + p.y * 0.00035), 0.0, 1.0);
        vec3 dayBase = mix(dayDeep, dayMid, midMask * 0.55);

        vec3 deepWater = mix(obsDeep, dayBase, uPhotorealMix);
        vec3 skyRefl = mix(obsRefl, dayRefl, uPhotorealMix);
        vec3 col = mix(deepWater, skyRefl, clamp(fresnel * 0.82 + 0.06, 0.0, 1.0));

        // Crisp sunlight wave specular glints (capped cleanly below daylight bloom threshold)
        vec3 L = normalize(uSunDir);
        vec3 H = normalize(L + V);
        float specTight = pow(max(dot(N, H), 0.0), 260.0) * (0.45 + 0.55 * rippleFade);
        float specBroad = pow(max(dot(N, H), 0.0), 56.0) * 0.14;
        col += uSunColor * (specTight * 0.78 + specBroad) * (0.30 + 0.70 * uDaylight);

        // Tactical 1km water grid ONLY in Tactical Obsidian mode
        if (uPhotorealMix < 0.98) {
          vec2 g = p / 1000.0;
          vec2 gd = abs(fract(g - 0.5) - 0.5) / max(fwidth(g), vec2(1e-4));
          float gl = (1.0 - smoothstep(0.0, 1.3, min(gd.x, gd.y))) * exp(-dist * 0.00018);
          col += vec3(0.08, 0.36, 0.55) * gl * 0.16 * (1.0 - uPhotorealMix);
        }

        float fogFactor = 1.0 - exp(-pow(dist * uFogDensity, 1.42));
        col = mix(col, uFogColor, clamp(fogFactor, 0.0, 0.96));
        col = min(col, vec3(0.88));
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });

  const geo = new THREE.PlaneGeometry(54000, 54000, 1, 1);
  geo.rotateX(-Math.PI * 0.5);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(1800, -3.35, -2000);
  mesh.receiveShadow = false;
  return { mesh, mat };
}

/** Builds the real 3D KSFO Airport Complex (runways, instanced ALSF-2 strobes, PAPI, ATC Tower, Terminals, Jetbridges, Hangars). */
function buildKsfoAirportComplex(): {
  readonly group: THREE.Group;
  readonly radarBar: THREE.Object3D;
  readonly airportBeaconRig: THREE.Object3D;
  readonly alsfStrobeInst: THREE.InstancedMesh;
  readonly alsfStrobeCount: number;
  readonly towerBeaconMat: THREE.MeshStandardMaterial;
} {
  const group = new THREE.Group();
  group.name = "ksfo_airport_complex";

  const fwdX = Math.sin(RWY_28_HDG_RAD);
  const fwdZ = -Math.cos(RWY_28_HDG_RAD);
  const rightX = Math.cos(RWY_28_HDG_RAD);
  const rightZ = Math.sin(RWY_28_HDG_RAD);

  // Crisp narrow taxiway strips connecting terminals to Runways 28L/28R/19L/19R (NO giant gray slab!)
  const taxiMat = new THREE.MeshStandardMaterial({
    color: 0x111a2a,
    roughness: 0.78,
    metalness: 0.18,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  const addTaxiStrip = (alongStart: number, alongEnd: number, rightOff: number, widthM: number) => {
    const len = Math.abs(alongEnd - alongStart);
    const mid = 0.5 * (alongStart + alongEnd);
    const geo = new THREE.PlaneGeometry(len, widthM);
    geo.rotateX(-Math.PI * 0.5);
    const mesh = new THREE.Mesh(geo, taxiMat);
    mesh.position.set(fwdX * mid + rightX * rightOff, 0.55, fwdZ * mid + rightZ * rightOff);
    mesh.rotation.y = Math.atan2(-fwdZ, fwdX);
    group.add(mesh);
  };
  addTaxiStrip(120, 3320, -145, 28);
  addTaxiStrip(880, 1920, -520, 140);

  // Helper to place a crisp textured runway starting at threshold `(thrX, thrZ)` heading `hdgRad`
  const addRunway = (
    thrX: number,
    thrZ: number,
    hdgRad: number,
    lengthM: number,
    widthM: number,
    yElev: number,
    labelA: string,
    labelB: string,
  ): void => {
    const tex = createRunwayCanvasTexture(labelA, labelB);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xcbd5e1,
      map: tex,
      roughness: 0.72,
      metalness: 0.12,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    });
    const geo = new THREE.PlaneGeometry(lengthM, widthM);
    geo.rotateX(-Math.PI * 0.5);
    const mesh = new THREE.Mesh(geo, mat);
    const fx = Math.sin(hdgRad);
    const fz = -Math.cos(hdgRad);
    mesh.position.set(thrX + fx * (lengthM * 0.5), yElev, thrZ + fz * (lengthM * 0.5));
    // Plane local +X aligns with runway heading (fx, 0, fz) -> rotation.y = atan2(-fz, fx)
    mesh.rotation.y = Math.atan2(-fz, fx);
    mesh.receiveShadow = true;
    group.add(mesh);
  };

  // Runways 28L (3470m x 61m) and parallel 28R (3618m x 61m, offset 229m starboard/NE)
  addRunway(0, 0, RWY_28_HDG_RAD, 3470, 61, 0.95, "28L", "10R");
  addRunway(rightX * 229 + fwdX * 110, rightZ * 229 + fwdZ * 110, RWY_28_HDG_RAD, 3618, 61, 0.97, "28R", "10L");

  // Crossing Runways 19L & 19R (real KSFO X-intersection layout!)
  const crossCx = fwdX * 1650 + rightX * 120;
  const crossCz = fwdZ * 1650 + rightZ * 120;
  const fx19 = Math.sin(RWY_19_HDG_RAD);
  const fz19 = -Math.cos(RWY_19_HDG_RAD);
  addRunway(crossCx - fx19 * 1150, crossCz - fz19 * 1150, RWY_19_HDG_RAD, 2332, 61, 0.99, "19R", "01L");
  addRunway(crossCx - fx19 * 1250 + fwdX * 230, crossCz - fz19 * 1250 + fwdZ * 230, RWY_19_HDG_RAD, 2636, 61, 1.01, "19L", "01R");

  // Glowing Runway 28L & 28R Edge Lights (InstancedMesh -> crisp HDR bloom!)
  const edgeBulbGeo = new THREE.SphereGeometry(1.15, 8, 6);
  const edgeWhiteMat = new THREE.MeshStandardMaterial({
    color: 0xe0f2fe,
    emissive: 0xbae6fd,
    emissiveIntensity: 2.6,
    toneMapped: false,
  });
  const edgePositions: THREE.Vector3[] = [];
  for (const rwyOffset of [0, 229]) {
    for (let d = 60; d <= 3420; d += 95) {
      for (const side of [-33, 33]) {
        edgePositions.push(
          new THREE.Vector3(
            fwdX * d + rightX * (rwyOffset + side),
            1.1,
            fwdZ * d + rightZ * (rwyOffset + side),
          ),
        );
      }
    }
  }
  const edgeInst = new THREE.InstancedMesh(edgeBulbGeo, edgeWhiteMat, edgePositions.length);
  const dummy = new THREE.Object3D();
  edgePositions.forEach((p, idx) => {
    dummy.position.copy(p);
    dummy.updateMatrix();
    edgeInst.setMatrixAt(idx, dummy.matrix);
  });
  group.add(edgeInst);

  // Green threshold light bar on 28L (InstancedMesh -> 1 draw call)
  const greenThrMat = new THREE.MeshStandardMaterial({
    color: 0x34d399,
    emissive: 0x10b981,
    emissiveIntensity: 3.4,
    toneMapped: false,
  });
  const greenThrInst = new THREE.InstancedMesh(edgeBulbGeo, greenThrMat, 8);
  for (let idx = 0; idx < 8; idx++) {
    const s = -28 + idx * 8;
    dummy.position.set(rightX * s, 1.2, rightZ * s);
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    greenThrInst.setMatrixAt(idx, dummy.matrix);
  }
  greenThrInst.instanceMatrix.needsUpdate = true;
  group.add(greenThrInst);

  // Animated Blooming ALSF-2 Approach Lighting System ("The Rabbit") over SF Bay (3 InstancedMeshes instead of 54 meshes!)
  const alsfStrobeCount = 18;
  const pilingMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.7, metalness: 0.4 });
  const alsfPylonGeo = new THREE.BoxGeometry(1.8, 4.8, 1.8);
  const alsfBarGeo = new THREE.BoxGeometry(16, 1.3, 1.6);
  const alsfStrobeGeo = new THREE.SphereGeometry(1.6, 10, 8);

  const alsfPylonInst = new THREE.InstancedMesh(alsfPylonGeo, pilingMat, alsfStrobeCount);
  const alsfBarMat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
  const alsfBarInst = new THREE.InstancedMesh(alsfBarGeo, alsfBarMat, alsfStrobeCount);
  const alsfStrobeMat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
  const alsfStrobeInst = new THREE.InstancedMesh(alsfStrobeGeo, alsfStrobeMat, alsfStrobeCount);

  const barRedHdr = new THREE.Color(2.4, 0.55, 0.55);
  const barWarmHdr = new THREE.Color(2.1, 1.95, 1.45);
  const strobeBaseHdr = new THREE.Color(0.9, 1.15, 1.35);

  for (let i = 0; i < alsfStrobeCount; i++) {
    const distBack = 55 + i * 48;
    const bx = -fwdX * distBack;
    const bz = -fwdZ * distBack;

    // Pier support pylon over SF Bay water
    dummy.position.set(bx, -1.6, bz);
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    alsfPylonInst.setMatrixAt(i, dummy.matrix);

    // Crossbar steady glow (1000-ft red decision bar at i === 6)
    dummy.position.set(bx, 0.85, bz);
    dummy.rotation.set(0, -RWY_28_HDG_RAD, 0);
    dummy.scale.set(i === 6 ? 2.4 : 1.0, 1.0, 1.0);
    dummy.updateMatrix();
    alsfBarInst.setMatrixAt(i, dummy.matrix);
    alsfBarInst.setColorAt(i, i === 6 ? barRedHdr : barWarmHdr);

    // Sequential running white strobe rabbit
    dummy.position.set(bx, 1.6, bz);
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    alsfStrobeInst.setMatrixAt(i, dummy.matrix);
    alsfStrobeInst.setColorAt(i, strobeBaseHdr);
  }
  alsfPylonInst.instanceMatrix.needsUpdate = true;
  alsfBarInst.instanceMatrix.needsUpdate = true;
  if (alsfBarInst.instanceColor) alsfBarInst.instanceColor.needsUpdate = true;
  alsfStrobeInst.instanceMatrix.needsUpdate = true;
  if (alsfStrobeInst.instanceColor) alsfStrobeInst.instanceColor.needsUpdate = true;
  group.add(alsfPylonInst);
  group.add(alsfBarInst);
  group.add(alsfStrobeInst);

  // Live 4-Light PAPI on port side of 28L (310 m down runway, 1 InstancedMesh)
  const papiGeo = new THREE.BoxGeometry(3.4, 1.8, 3.4);
  const papiMat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
  const papiInst = new THREE.InstancedMesh(papiGeo, papiMat, 4);
  const papiWhiteHdr = new THREE.Color(3.6, 3.5, 2.8);
  const papiRedHdr = new THREE.Color(3.6, 0.55, 0.45);
  for (let p = 0; p < 4; p++) {
    dummy.position.set(
      fwdX * 310 - rightX * (54 + p * 8.5),
      1.6,
      fwdZ * 310 - rightZ * (54 + p * 8.5),
    );
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    papiInst.setMatrixAt(p, dummy.matrix);
    papiInst.setColorAt(p, p < 2 ? papiWhiteHdr : papiRedHdr);
  }
  papiInst.instanceMatrix.needsUpdate = true;
  if (papiInst.instanceColor) papiInst.instanceColor.needsUpdate = true;
  group.add(papiInst);

  // Iconic 3D KSFO Air Traffic Control Tower (south-west of terminal core, ~72 m tall!)
  const towerGroup = new THREE.Group();
  const towerX = fwdX * 1420 - rightX * 680;
  const towerZ = fwdZ * 1420 - rightZ * 680;
  towerGroup.position.set(towerX, 0.3, towerZ);

  const towerSteelMat = new THREE.MeshStandardMaterial({
    color: 0xdbe5f0,
    roughness: 0.28,
    metalness: 0.58,
  });
  const towerObsidianMat = new THREE.MeshStandardMaterial({
    color: 0x0f1a2b,
    roughness: 0.35,
    metalness: 0.65,
  });
  const towerGlassMat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    emissive: 0x0ea5e9,
    emissiveIntensity: 2.1,
    roughness: 0.1,
    metalness: 0.4,
    toneMapped: false,
  });

  // Sculpted hyperboloid concrete/steel shaft (KSFO's real twisting architectural tower silhouette)
  const shaftPts: THREE.Vector2[] = [
    new THREE.Vector2(15.0, 0),
    new THREE.Vector2(10.5, 16),
    new THREE.Vector2(7.6, 36),
    new THREE.Vector2(8.8, 52),
    new THREE.Vector2(13.8, 60),
    new THREE.Vector2(15.2, 62),
  ];
  const shaft = new THREE.Mesh(new THREE.LatheGeometry(shaftPts, 32), towerSteelMat);
  shaft.castShadow = true;
  towerGroup.add(shaft);

  // Glowing cyan 360-degree glass observation cab
  const cab = new THREE.Mesh(new THREE.CylinderGeometry(15.6, 13.4, 8.2, 24), towerGlassMat);
  cab.position.y = 66.2;
  towerGroup.add(cab);

  // Cantilevered architectural roof canopy
  const roof = new THREE.Mesh(new THREE.CylinderGeometry(18.5, 16.5, 2.4, 24), towerObsidianMat);
  roof.position.y = 71.2;
  towerGroup.add(roof);

  // Continuously rotating ASR rooftop radar bar
  const radarBar = new THREE.Group();
  radarBar.position.y = 75.5;
  const radarPedestal = new THREE.Mesh(new THREE.CylinderGeometry(1.4, 2.0, 4.5, 12), towerSteelMat);
  radarPedestal.position.y = -1.5;
  radarBar.add(radarPedestal);
  const radarBarMesh = new THREE.Mesh(new THREE.BoxGeometry(11.5, 1.5, 1.2), towerSteelMat);
  radarBar.add(radarBarMesh);
  towerGroup.add(radarBar);

  // Rotating Civil Airport Beacon (alternating White / Emerald-Green dual lens atop ATC Tower!)
  const airportBeaconRig = new THREE.Group();
  airportBeaconRig.position.y = 79.4;
  const beaconGreen = new THREE.Mesh(
    new THREE.SphereGeometry(1.45, 10, 8),
    new THREE.MeshStandardMaterial({
      color: 0x34d399,
      emissive: 0x10b981,
      emissiveIntensity: 4.2,
      toneMapped: false,
    }),
  );
  beaconGreen.position.x = 2.0;
  airportBeaconRig.add(beaconGreen);
  const beaconWhite = new THREE.Mesh(
    new THREE.SphereGeometry(1.45, 10, 8),
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xe0f2fe,
      emissiveIntensity: 4.2,
      toneMapped: false,
    }),
  );
  beaconWhite.position.x = -2.0;
  airportBeaconRig.add(beaconWhite);
  towerGroup.add(airportBeaconRig);

  const towerBeaconMat = new THREE.MeshStandardMaterial({
    color: 0xff3b30,
    emissive: 0xef4444,
    emissiveIntensity: 4.0,
    toneMapped: false,
  });
  const towerBeacon = new THREE.Mesh(new THREE.SphereGeometry(1.1, 10, 8), towerBeaconMat);
  towerBeacon.position.y = 82.0;
  towerGroup.add(towerBeacon);
  group.add(towerGroup);

  // 3D KSFO Terminal Concourses & Boarding Piers (modern dark steel + warm/cyan lit glass)
  const termBodyMat = new THREE.MeshStandardMaterial({
    color: 0x162235,
    roughness: 0.38,
    metalness: 0.62,
  });
  const termRoofMat = new THREE.MeshStandardMaterial({
    color: 0x263852,
    roughness: 0.34,
    metalness: 0.72,
  });
  const termWindowWarmMat = new THREE.MeshStandardMaterial({
    color: 0xfde68a,
    emissive: 0xf59e0b,
    emissiveIntensity: 1.65,
    toneMapped: false,
  });
  const termWindowCyanMat = new THREE.MeshStandardMaterial({
    color: 0x7dd3fc,
    emissive: 0x0ea5e9,
    emissiveIntensity: 1.4,
    toneMapped: false,
  });

  const terminalSpecs: Array<[number, number, number, number, number, number, boolean]> = [
    // [along28L, rightOffset, len, h, wid, rotDeg, warmGlass]
    [1360, -820, 380, 22, 95, 0, true], // International Terminal main hall
    [1180, -610, 290, 16, 48, 22, false], // Boarding Area G pier
    [1580, -560, 310, 16, 48, -28, true], // Boarding Area F pier
    [1760, -820, 280, 17, 56, 8, false], // Terminal 3 concourse
    [1020, -880, 290, 16, 52, -16, true], // Terminal 1 concourse
  ];

  // Sculpted passenger boarding jetbridges batched into a single InstancedMesh (15 -> 1 draw call)
  const jetbridgeGeo = new THREE.BoxGeometry(9.5, 4.2, 34);
  const jetbridgeInst = new THREE.InstancedMesh(jetbridgeGeo, termRoofMat, terminalSpecs.length * 3);
  const jbParentMat = new THREE.Matrix4();
  const jbLocalMat = new THREE.Matrix4();
  const jbFinalMat = new THREE.Matrix4();
  const jbQuat = new THREE.Quaternion();
  const jbPos = new THREE.Vector3();
  const jbScale = new THREE.Vector3(1, 1, 1);
  const jbLocalPos = new THREE.Vector3();
  const jbIdentityQ = new THREE.Quaternion();
  let jbIdx = 0;

  for (const [along, rightOff, len, h, wid, rotDeg, isWarm] of terminalSpecs) {
    const tGroup = new THREE.Group();
    const tx = fwdX * along + rightX * rightOff;
    const ty = h * 0.5;
    const tz = fwdZ * along + rightZ * rightOff;
    const rotY = -RWY_28_HDG_RAD + Math.PI * 0.5 + rotDeg * DEG_TO_RAD;
    tGroup.position.set(tx, ty, tz);
    tGroup.rotation.y = rotY;

    const mainBox = new THREE.Mesh(new THREE.BoxGeometry(len, h, wid), termBodyMat);
    mainBox.castShadow = true;
    tGroup.add(mainBox);

    // Sculpted overhanging roof slab
    const roofSlab = new THREE.Mesh(new THREE.BoxGeometry(len + 10, 2.2, wid + 12), termRoofMat);
    roofSlab.position.y = h * 0.5 + 1.1;
    tGroup.add(roofSlab);

    // Illuminated glass curtain-wall band
    const glassBand = new THREE.Mesh(
      new THREE.BoxGeometry(len + 1.2, h * 0.34, wid + 1.2),
      isWarm ? termWindowWarmMat : termWindowCyanMat,
    );
    glassBand.position.y = h * 0.1;
    tGroup.add(glassBand);

    jbPos.set(tx, ty, tz);
    jbQuat.setFromAxisAngle(new THREE.Vector3(0, 1, 0), rotY);
    jbParentMat.compose(jbPos, jbQuat, jbScale);
    for (let j = -1; j <= 1; j++) {
      jbLocalPos.set(j * (len * 0.28), -h * 0.12, wid * 0.5 + 16);
      jbLocalMat.compose(jbLocalPos, jbIdentityQ, jbScale);
      jbFinalMat.multiplyMatrices(jbParentMat, jbLocalMat);
      jetbridgeInst.setMatrixAt(jbIdx++, jbFinalMat);
    }

    group.add(tGroup);
  }
  jetbridgeInst.instanceMatrix.needsUpdate = true;
  group.add(jetbridgeInst);

  // 3D Aircraft Maintenance Hangars (arched steel roofs south of Taxiway C)
  const hangarMat = new THREE.MeshStandardMaterial({
    color: 0x2a394e,
    roughness: 0.42,
    metalness: 0.58,
  });
  const hangarArchGeo = new THREE.CylinderGeometry(42, 42, 115, 16, 1, false, 0, Math.PI);
  hangarArchGeo.rotateZ(Math.PI * 0.5);
  for (const [hAlong, hRight] of [
    [2240, -480],
    [2440, -520],
    [2640, -560],
  ] as const) {
    const hGroup = new THREE.Group();
    hGroup.position.set(fwdX * hAlong + rightX * hRight, 1.0, fwdZ * hAlong + rightZ * hRight);
    hGroup.rotation.y = -RWY_28_HDG_RAD + Math.PI * 0.5;
    const base = new THREE.Mesh(new THREE.BoxGeometry(115, 18, 84), termBodyMat);
    base.position.y = 9;
    hGroup.add(base);
    const roofArch = new THREE.Mesh(hangarArchGeo, hangarMat);
    roofArch.position.y = 18;
    roofArch.scale.set(1.0, 0.32, 1.0);
    hGroup.add(roofArch);
    group.add(hGroup);
  }

  return { group, radarBar, airportBeaconRig, alsfStrobeInst, alsfStrobeCount, towerBeaconMat };
}

/** Builds 3D San Francisco Downtown Skyline + Bay Bridge suspension towers on the horizon. */
function buildSfSkylineAndBridge(): THREE.Group {
  const group = new THREE.Group();
  group.name = "sf_skyline_and_bridge";

  // 1. 3D Downtown San Francisco Skyline (~14 km north of KSFO)
  const bldgCount = 175;
  const bldgGeo = new THREE.BoxGeometry(1, 1, 1);
  bldgGeo.translate(0, 0.5, 0);
  const bldgMat = new THREE.MeshStandardMaterial({
    color: 0x0f1a2d,
    roughness: 0.35,
    metalness: 0.65,
    emissive: 0x1e3a5f,
    emissiveIntensity: 0.35,
  });
  const skylineInst = new THREE.InstancedMesh(bldgGeo, bldgMat, bldgCount);

  // Glowing architectural rooftop / spire crown lights
  const crownGeo = new THREE.BoxGeometry(1, 1, 1);
  const crownMat = new THREE.MeshStandardMaterial({
    color: 0xfef3c7,
    emissive: 0x38bdf8,
    emissiveIntensity: 1.85,
    toneMapped: false,
  });
  const crownInst = new THREE.InstancedMesh(crownGeo, crownMat, 60);

  const dummy = new THREE.Object3D();
  let crownIdx = 0;
  for (let i = 0; i < bldgCount; i++) {
    // Deterministic pseudo-random cluster centered on Downtown SF (-1800 E, 13800 N -> z = -13800)
    const s1 = Math.sin(i * 12.9898 + 1.7) * 43758.5453;
    const s2 = Math.sin(i * 78.233 + 3.1) * 24634.6345;
    const s3 = Math.sin(i * 39.425 + 5.9) * 19531.2312;
    const r1 = s1 - Math.floor(s1);
    const r2 = s2 - Math.floor(s2);
    const r3 = s3 - Math.floor(s3);

    const angle = r1 * Math.PI * 2;
    const rad = Math.pow(r2, 0.72) * 2200.0;
    const x = -1800 + Math.cos(angle) * rad * 1.35;
    const z = -13600 + Math.sin(angle) * rad * 0.75;
    const centerBoost = Math.exp(-(rad * rad) / (2 * 820 * 820));
    const h = 42 + r3 * 165 + centerBoost * 155;
    const w = 34 + (1.0 - r3) * 44;

    dummy.position.set(x, 4.0, z);
    dummy.scale.set(w, h, w);
    dummy.rotation.y = (i % 4) * 0.35;
    dummy.updateMatrix();
    skylineInst.setMatrixAt(i, dummy.matrix);

    if (crownIdx < 60 && h > 135) {
      dummy.position.set(x, h + 5.0, z);
      dummy.scale.set(w * 0.78, 3.2, w * 0.78);
      dummy.updateMatrix();
      crownInst.setMatrixAt(crownIdx++, dummy.matrix);
    }
  }
  crownInst.count = crownIdx;
  group.add(skylineInst);
  group.add(crownInst);

  // Iconic Salesforce Tower obelisk + glowing crown & Transamerica Pyramid
  const spireMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    roughness: 0.22,
    metalness: 0.78,
  });
  const salesforce = new THREE.Mesh(new THREE.CylinderGeometry(24, 42, 345, 20), spireMat);
  salesforce.position.set(-1620, 175, -13550);
  group.add(salesforce);

  const sfCrown = new THREE.Mesh(
    new THREE.CylinderGeometry(22, 26, 26, 20),
    new THREE.MeshStandardMaterial({
      color: 0xe0f2fe,
      emissive: 0x38bdf8,
      emissiveIntensity: 2.5,
      toneMapped: false,
    }),
  );
  sfCrown.position.set(-1620, 336, -13550);
  group.add(sfCrown);

  const transamerica = new THREE.Mesh(new THREE.ConeGeometry(44, 280, 4), spireMat);
  transamerica.position.set(-2240, 144, -14100);
  group.add(transamerica);

  // 2. 3D Bay Suspension Bridge Towers spanning East across SF Bay
  const bridgeMat = new THREE.MeshStandardMaterial({
    color: 0x334155,
    roughness: 0.38,
    metalness: 0.72,
  });
  const bridgeLightMat = new THREE.MeshStandardMaterial({
    color: 0xfef08a,
    emissive: 0xf59e0b,
    emissiveIntensity: 2.6,
    toneMapped: false,
  });
  const deckGeo = new THREE.BoxGeometry(9200, 5.5, 28);
  const deck = new THREE.Mesh(deckGeo, bridgeMat);
  deck.position.set(4200, 52, -10800);
  deck.rotation.y = 0.24;
  group.add(deck);

  const pylonInst = new THREE.InstancedMesh(new THREE.BoxGeometry(14, 185, 14), bridgeMat, 10);
  const crossInst = new THREE.InstancedMesh(new THREE.BoxGeometry(12, 10, 44), bridgeMat, 5);
  const lampInst = new THREE.InstancedMesh(new THREE.SphereGeometry(3.2, 8, 6), bridgeLightMat, 5);
  for (let t = 0; t < 5; t++) {
    const tx = 650 + t * 1750;
    const tz = -11650 + t * 420;
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, 1, 1);

    dummy.position.set(tx, 90, tz - 16);
    dummy.updateMatrix();
    pylonInst.setMatrixAt(t * 2, dummy.matrix);

    dummy.position.set(tx, 90, tz + 16);
    dummy.updateMatrix();
    pylonInst.setMatrixAt(t * 2 + 1, dummy.matrix);

    dummy.position.set(tx, 165, tz);
    dummy.updateMatrix();
    crossInst.setMatrixAt(t, dummy.matrix);

    dummy.position.set(tx, 186, tz);
    dummy.updateMatrix();
    lampInst.setMatrixAt(t, dummy.matrix);
  }
  pylonInst.instanceMatrix.needsUpdate = true;
  crossInst.instanceMatrix.needsUpdate = true;
  lampInst.instanceMatrix.needsUpdate = true;
  group.add(pylonInst);
  group.add(crossInst);
  group.add(lampInst);

  return group;
}

/** Builds the complete Dual-Aesthetic 3D KSFO Digital Twin World. */
export function createKsfoWorld(initialPack: KsfoTerrainPack | null = null): KsfoWorldRig {
  const group = new THREE.Group();
  group.name = "ksfo_digital_twin_world";

  let visualMode: VisualWorldMode = "photoreal";
  const terrainMaterial = createObsidianTerrainMaterial();

  // Inner 256x256 (10km x 10km) real USGS 3DEP terrain centered on KSFO + San Bruno Mountain
  const innerGeo = buildTerrainGeometry(10000, 256, INNER10K_CX, INNER10K_CZ, initialPack);
  const innerMesh = new THREE.Mesh(innerGeo, terrainMaterial);
  innerMesh.position.set(INNER10K_CX, 0, INNER10K_CZ);
  innerMesh.receiveShadow = true;
  group.add(innerMesh);

  // Outer 128x128 (34km x 34km) surrounding Bay Area mountains (shifted slightly down to avoid z-fight)
  const outerGeo = buildTerrainGeometry(34000, 128, INNER10K_CX, INNER10K_CZ, initialPack);
  const outerMesh = new THREE.Mesh(outerGeo, terrainMaterial);
  outerMesh.position.set(INNER10K_CX, -1.8, INNER10K_CZ);
  group.add(outerMesh);

  // Realistic Animated SF Bay Water
  const water = createSfBayWaterMesh();
  group.add(water.mesh);

  // Real 3D KSFO Airport Complex (Runways, ALSF-2, PAPI, ATC Tower, Terminals, Jetbridges, Hangars)
  const airport = buildKsfoAirportComplex();
  group.add(airport.group);

  // 3D SF Downtown Skyline + Bay Bridge
  const skyline = buildSfSkylineAndBridge();
  group.add(skyline);

  const alsfStrobeCol = new THREE.Color();

  return {
    group,
    terrainMaterial,
    setTerrainPack(pack: KsfoTerrainPack): void {
      displaceTerrainGeometry(innerGeo, INNER10K_CX, INNER10K_CZ, pack);
      displaceTerrainGeometry(outerGeo, INNER10K_CX, INNER10K_CZ, pack);
    },
    setOrthoTextures(core2k: THREE.Texture | null, inner10k: THREE.Texture | null): void {
      if (core2k) {
        core2k.colorSpace = THREE.SRGBColorSpace;
        core2k.anisotropy = 8;
        terrainMaterial.uniforms.uCoreOrtho.value = core2k;
        terrainMaterial.uniforms.uHasCoreOrtho.value = 1.0;
      }
      if (inner10k) {
        inner10k.colorSpace = THREE.SRGBColorSpace;
        inner10k.anisotropy = 8;
        terrainMaterial.uniforms.uInnerOrtho.value = inner10k;
        terrainMaterial.uniforms.uHasInnerOrtho.value = 1.0;
      }
    },
    setVisualMode(mode: VisualWorldMode): void {
      visualMode = mode;
      const mixVal = mode === "photoreal" ? 1.0 : 0.0;
      terrainMaterial.uniforms.uPhotorealMix.value = mixVal;
      water.mat.uniforms.uPhotorealMix.value = mixVal;
    },
    getVisualMode(): VisualWorldMode {
      return visualMode;
    },
    update(simTimeS: number, lighting: AtmosphereLightingState, fogDensity: number): void {
      water.mat.uniforms.uTime.value = simTimeS;
      water.mat.uniforms.uSunDir.value.copy(lighting.sunDir);
      water.mat.uniforms.uSunColor.value.copy(lighting.sunColor);
      water.mat.uniforms.uDaylight.value = lighting.daylightFactor;
      water.mat.uniforms.uFogColor.value.copy(lighting.fogColor);
      water.mat.uniforms.uFogDensity.value = fogDensity;

      terrainMaterial.uniforms.uSunDir.value.copy(lighting.sunDir);
      terrainMaterial.uniforms.uSunColor.value.copy(lighting.sunColor);
      terrainMaterial.uniforms.uDaylight.value = lighting.daylightFactor;
      terrainMaterial.uniforms.uFogColor.value.copy(lighting.fogColor);
      terrainMaterial.uniforms.uFogDensity.value = fogDensity;

      // Rotate KSFO ATC Tower ASR radar + alternating green/white civil airport beacon
      airport.radarBar.rotation.y = simTimeS * 1.45;
      airport.airportBeaconRig.rotation.y = simTimeS * 2.1;
      airport.towerBeaconMat.emissiveIntensity =
        1.2 + 3.4 * Math.pow(0.5 + 0.5 * Math.sin(simTimeS * 3.8), 4.0);

      // Animate instanced ALSF-2 running white strobe rabbit racing from outer bay -> 28L threshold at 2 Hz!
      const rabbitPos = (1.0 - ((simTimeS * 1.85) % 1.0)) * (airport.alsfStrobeCount - 1);
      for (let i = 0; i < airport.alsfStrobeCount; i++) {
        const d = Math.abs(i - rabbitPos);
        const flash = Math.exp(-d * d * 1.6);
        const intensity = 0.55 + 4.8 * flash;
        alsfStrobeCol.setRGB(intensity * 0.85, intensity * 0.96, intensity);
        airport.alsfStrobeInst.setColorAt(i, alsfStrobeCol);
      }
      if (airport.alsfStrobeInst.instanceColor) {
        airport.alsfStrobeInst.instanceColor.needsUpdate = true;
      }
    },
  };
}
