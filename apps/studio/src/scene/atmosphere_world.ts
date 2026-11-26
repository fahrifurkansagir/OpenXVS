/**
 * Photorealistic Physical Atmosphere + Sunlit White Multi-Puff 3D Cumulus Cloud Deck (`OpenXVS Studio v5.0`).
 *
 * - Rich Coastal California Blue Zenith (#1b5ca8) fading naturally to bright sunlit horizon (#d6e8f7)
 *   with warm afternoon golden-hour glow (#f5b870) near the low western sun + subtle solar disc/halo.
 * - Multi-puff volumetric white cumulus clouds (#ffffff sunlit tops, #9ab2cc cool sky undersides)
 *   drifting smoothly with live crosswind.
 * - Dual aesthetic: `"photoreal"` (default Daylight / Golden Hour 16.5h) vs `"obsidian"` (Tactical Obsidian).
 */

import * as THREE from "three";
import type { WeatherState } from "../sim/studio_sim.ts";

export type VisualWorldMode = "photoreal" | "obsidian";

export interface AtmosphereLightingState {
  readonly sunDir: THREE.Vector3;
  readonly sunElevationDeg: number;
  readonly daylightFactor: number; // 0 = night/obsidian, 1 = bright daylight
  readonly goldenFactor: number; // peaks around sunrise/sunset
  readonly sunColor: THREE.Color;
  readonly sunIntensity: number;
  readonly hemiSkyColor: THREE.Color;
  readonly hemiGroundColor: THREE.Color;
  readonly hemiIntensity: number;
  readonly fogColor: THREE.Color;
}

export interface AtmosphereRig {
  readonly group: THREE.Group;
  readonly skyMesh: THREE.Mesh;
  readonly cloudsMesh: THREE.InstancedMesh;
  computeLighting(timeOfDayH: number, mode: VisualWorldMode, cloudCover: number): AtmosphereLightingState;
  update(
    simTimeS: number,
    dtS: number,
    cameraPos: THREE.Vector3,
    weather: WeatherState,
    mode: VisualWorldMode,
  ): AtmosphereLightingState;
}

const DEG_TO_RAD = Math.PI / 180.0;

// Hoisted module-level color constants (zero per-frame allocations)
const COL_NIGHT_SUN = new THREE.Color(0x64748b);
const COL_GOLD_SUN = new THREE.Color(0xffd29d);
const COL_NOON_SUN = new THREE.Color(0xfffaf2);
const COL_NIGHT_SKY = new THREE.Color(0x13294b);
const COL_DAY_SKY = new THREE.Color(0x9bd0ff);
const COL_NIGHT_GND = new THREE.Color(0x0b1526);
const COL_DAY_GND = new THREE.Color(0x3b4d5e);
const COL_NIGHT_FOG = new THREE.Color(0x091628);
const COL_DAY_FOG = new THREE.Color(0xc9e0f2);
const COL_GOLD_FOG = new THREE.Color(0xdccbb8);
const COL_OVERCAST_FOG = new THREE.Color(0xb8c7d4);
const COL_CLOUD_NIGHT_TOP = new THREE.Color(0x6c85a6);
const COL_CLOUD_DAY_TOP = new THREE.Color(0xffffff);
const COL_CLOUD_GOLD_TOP = new THREE.Color(0xfff1de);
const COL_CLOUD_NIGHT_BOT = new THREE.Color(0x0c1729);
const COL_CLOUD_DAY_BOT = new THREE.Color(0x8ea9c7);

interface MutableAtmosphereLightingState {
  sunDir: THREE.Vector3;
  sunElevationDeg: number;
  daylightFactor: number;
  goldenFactor: number;
  sunColor: THREE.Color;
  sunIntensity: number;
  hemiSkyColor: THREE.Color;
  hemiGroundColor: THREE.Color;
  hemiIntensity: number;
  fogColor: THREE.Color;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

function computeSolarAtmosphereInto(
  timeOfDayH: number,
  mode: VisualWorldMode,
  cloudCover: number,
  out: MutableAtmosphereLightingState,
): AtmosphereLightingState {
  const h = clamp(timeOfDayH, 5.0, 22.0);
  // San Francisco (37.6°N) solar elevation curve: noon ~52°, golden afternoon 16.5h ~18.5°, sunset ~18.4h
  const dayPhase = ((h - 6.4) / 12.0) * Math.PI;
  const rawElevRad = Math.sin(dayPhase) * (52.0 * DEG_TO_RAD) - 0.03;
  const azRad = (84.0 + ((h - 6.4) / 12.0) * 196.0) * DEG_TO_RAD;

  const elevRad = mode === "obsidian" ? 6.0 * DEG_TO_RAD : rawElevRad;
  if (mode === "obsidian") {
    out.sunDir.set(-0.82, 0.12, -0.55).normalize();
  } else {
    out.sunDir
      .set(
        Math.cos(elevRad) * Math.sin(azRad),
        Math.sin(elevRad),
        -Math.cos(elevRad) * Math.cos(azRad),
      )
      .normalize();
  }

  const elevDeg = elevRad / DEG_TO_RAD;
  const daylightFactor =
    mode === "obsidian" ? 0.0 : smoothstep(-3.0, 14.0, elevDeg);
  const goldenFactor =
    mode === "obsidian"
      ? 0.0
      : smoothstep(-4.0, 6.0, elevDeg) * (1.0 - smoothstep(18.0, 38.0, elevDeg));

  const overcastDamp = 1.0 - 0.28 * clamp(cloudCover, 0.0, 1.0);

  if (mode === "obsidian") {
    out.sunColor.setHex(0xffedd5);
    out.hemiSkyColor.setHex(0x7dd3fc);
    out.hemiGroundColor.setHex(0x0b1526);
    out.fogColor.setHex(0x091628);
  } else {
    out.sunColor
      .copy(COL_NIGHT_SUN)
      .lerp(COL_NOON_SUN, daylightFactor)
      .lerp(COL_GOLD_SUN, goldenFactor * 0.55);

    out.hemiSkyColor.copy(COL_NIGHT_SKY).lerp(COL_DAY_SKY, daylightFactor);
    out.hemiGroundColor.copy(COL_NIGHT_GND).lerp(COL_DAY_GND, daylightFactor);

    // Sunlit coastal horizon fog (#c9e0f2) warming subtly toward #e8cfb4 at golden hour
    out.fogColor
      .copy(COL_NIGHT_FOG)
      .lerp(COL_DAY_FOG, daylightFactor)
      .lerp(COL_GOLD_FOG, goldenFactor * 0.34)
      .lerp(COL_OVERCAST_FOG, cloudCover * 0.25 * daylightFactor);
  }

  out.sunElevationDeg = elevDeg;
  out.daylightFactor = daylightFactor;
  out.goldenFactor = goldenFactor;
  out.sunIntensity =
    mode === "obsidian"
      ? 2.25
      : (0.45 + 2.75 * daylightFactor) * overcastDamp;
  out.hemiIntensity =
    mode === "obsidian" ? 0.92 : 0.45 + 0.85 * daylightFactor;

  return out;
}

/** Computes realistic San Francisco (37.6°N) solar vector and crisp coastal daylight/golden-hour lighting. */
export function computeSolarAtmosphere(
  timeOfDayH: number,
  mode: VisualWorldMode,
  cloudCover: number,
): AtmosphereLightingState {
  return computeSolarAtmosphereInto(timeOfDayH, mode, cloudCover, {
    sunDir: new THREE.Vector3(),
    sunElevationDeg: 0,
    daylightFactor: 0,
    goldenFactor: 0,
    sunColor: new THREE.Color(),
    sunIntensity: 0,
    hemiSkyColor: new THREE.Color(),
    hemiGroundColor: new THREE.Color(),
    hemiIntensity: 0,
    fogColor: new THREE.Color(),
  });
}

/** Rich Coastal California Rayleigh + Mie Physical Sky Dome (zero milky washout). */
function createPhysicalSkyDome(): { readonly mesh: THREE.Mesh; readonly mat: THREE.ShaderMaterial } {
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      uSunDir: { value: new THREE.Vector3(-0.78, 0.32, -0.52).normalize() },
      uDaylight: { value: 1.0 },
      uGolden: { value: 0.45 },
      uObsidian: { value: 0.0 },
      uCloudCover: { value: 0.42 },
      uTime: { value: 0.0 },
    },
    vertexShader: `
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 uSunDir;
      uniform float uDaylight;
      uniform float uGolden;
      uniform float uObsidian;
      uniform float uCloudCover;
      uniform float uTime;
      varying vec3 vDir;

      float hash21(vec2 p) {
        p = fract(p * vec2(123.34, 456.21));
        p += dot(p, p + 45.32);
        return fract(p.x * p.y);
      }

      void main() {
        vec3 d = normalize(vDir);
        vec3 sunD = normalize(uSunDir);
        float elev = clamp(d.y, -0.1, 1.0);
        float cosTheta = max(dot(d, sunD), 0.0);

        // Rich coastal California blue zenith (#1b5ca8 linear) -> vibrant sky -> bright sunlit horizon (#d6e8f7)
        vec3 dayZenith = vec3(0.025, 0.135, 0.44);
        vec3 dayMid = vec3(0.11, 0.34, 0.72);
        vec3 dayHorizon = vec3(0.58, 0.74, 0.88);

        vec3 nightZenith = vec3(0.014, 0.028, 0.064);
        vec3 nightMid = vec3(0.042, 0.098, 0.195);
        vec3 nightHorizon = vec3(0.092, 0.175, 0.295);

        vec3 zenith = mix(nightZenith, dayZenith, uDaylight);
        vec3 midSky = mix(nightMid, dayMid, uDaylight);
        vec3 horizon = mix(nightHorizon, dayHorizon, uDaylight);

        // Gentle marine layer softening only at high cloud cover
        horizon = mix(horizon, vec3(0.55, 0.65, 0.75), uCloudCover * 0.24 * uDaylight);

        vec3 col = mix(horizon, midSky, smoothstep(0.0, 0.18, max(elev, 0.0)));
        col = mix(col, zenith, smoothstep(0.16, 0.68, max(elev, 0.0)));

        // Warm golden-hour glow (#f5b870) near the low afternoon western sun
        float sunAzAlign = max(dot(normalize(vec3(d.x, 0.0, d.z)), normalize(vec3(sunD.x, 0.0, sunD.z))), 0.0);
        float horizonBand = exp(-abs(elev - 0.012) * 10.5);
        vec3 goldenGlow = vec3(0.88, 0.48, 0.16) * pow(sunAzAlign, 2.4) * horizonBand * (0.18 + 0.56 * uGolden) * (1.0 - uObsidian * 0.72);
        col += goldenGlow;

        // Crisp subtle solar disc + Mie aureole (strictly below daylight bloom threshold 0.92!)
        float mieHalo = pow(cosTheta, 22.0) * (0.20 + 0.24 * uGolden) * (1.0 - uObsidian * 0.78);
        float sunDisc = smoothstep(0.9988, 0.9997, cosTheta) * 0.52 * uDaylight * (1.0 - uCloudCover * 0.55);
        col += vec3(1.0, 0.88, 0.66) * (mieHalo + sunDisc);

        // Delicate high-altitude white cirrus streaks
        if (elev > 0.04) {
          vec2 cUv = d.xz / (elev + 0.20);
          float c = sin(cUv.x * 1.55 + uTime * 0.01) * cos(cUv.y * 1.95 - cUv.x * 0.65);
          float c2 = sin(cUv.x * 3.4 - cUv.y * 2.5 + uTime * 0.015) * 0.5 + 0.5;
          float cloudMask = smoothstep(0.38, 0.90, c * 0.62 + c2 * 0.38)
                          * smoothstep(0.04, 0.18, elev)
                          * (1.0 - smoothstep(0.45, 0.78, elev))
                          * (0.22 + 0.78 * uCloudCover);
          vec3 cirrusCol = mix(
            mix(vec3(0.16, 0.28, 0.42), vec3(0.86, 0.90, 0.95), uDaylight),
            vec3(0.92, 0.78, 0.62),
            pow(sunAzAlign, 2.5) * uGolden * 0.6
          );
          col = mix(col, cirrusCol, cloudMask * 0.26);
        }

        // Crisp stars at night / Tactical Obsidian
        float starVis = clamp(1.0 - uDaylight * 1.4, 0.0, 1.0);
        if (starVis > 0.05 && elev > 0.15) {
          vec2 sGrid = floor(d.xz * 260.0 / (elev + 0.35));
          float rnd = hash21(sGrid);
          float star = step(0.9965, rnd) * smoothstep(0.15, 0.52, elev) * starVis;
          col += vec3(0.75, 0.85, 1.0) * star * 0.58;
        }

        float maxSky = mix(0.88, 0.60, uObsidian);
        col = min(col, vec3(maxSky));
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });

  const mesh = new THREE.Mesh(new THREE.SphereGeometry(34000, 36, 24), mat);
  mesh.frustumCulled = false;
  return { mesh, mat };
}

interface CloudPuffSeed {
  x: number;
  y: number;
  z: number;
  sx: number;
  sy: number;
  sz: number;
  rotY: number;
  densityThreshold: number;
}

/** Sunlit White Multi-Puff 3D Cumulus Cloud Deck (stacked billowy domes, bright white tops, cool blue bases). */
function createVolumetricCloudDeck(): {
  readonly mesh: THREE.InstancedMesh;
  readonly mat: THREE.ShaderMaterial;
  readonly seeds: readonly CloudPuffSeed[];
} {
  const clusterCount = 38;
  const puffsPerCluster = 9;
  const totalPuffs = clusterCount * puffsPerCluster;

  const seeds: CloudPuffSeed[] = [];
  const rand = (n: number) => {
    const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
    return s - Math.floor(s);
  };

  for (let c = 0; c < clusterCount; c++) {
    const angle = rand(c * 3.1 + 1.3) * Math.PI * 2;
    const radius = 2600.0 + Math.pow(rand(c * 5.7 + 2.1), 0.72) * 14500.0;
    const cx = Math.cos(angle) * radius;
    const cz = Math.sin(angle) * radius - 1800.0;
    const cy = 760.0 + rand(c * 7.3 + 3.7) * 1100.0;
    const clusterThresh = c / clusterCount;

    // Build authentic cumulus silhouette: flat broad base puffs + taller central billowy crown domes
    for (let p = 0; p < puffsPerCluster; p++) {
      const idx = c * puffsPerCluster + p;
      const isCrown = p < 3;
      const pAng = (p / puffsPerCluster) * Math.PI * 2 + rand(idx * 4.1) * 0.7;
      const pRad = isCrown ? rand(idx * 8.3) * 190.0 : 180.0 + rand(idx * 11.3) * 320.0;
      const ox = Math.cos(pAng) * pRad * 1.15;
      const oz = Math.sin(pAng) * pRad * 0.95;
      const oy = isCrown ? 65.0 + rand(idx * 17.9) * 95.0 : (rand(idx * 15.1) - 0.3) * 38.0;
      const baseScale = isCrown ? 200.0 + rand(idx * 19.1) * 180.0 : 170.0 + rand(idx * 21.7) * 210.0;
      seeds.push({
        x: cx + ox,
        y: cy + oy,
        z: cz + oz,
        sx: baseScale * (1.1 + rand(idx * 23.3) * 0.45),
        sy: baseScale * (isCrown ? 0.68 + rand(idx * 29.7) * 0.25 : 0.42 + rand(idx * 29.7) * 0.18),
        sz: baseScale * (1.0 + rand(idx * 31.1) * 0.45),
        rotY: rand(idx * 37.3) * Math.PI * 2,
        densityThreshold: clusterThresh,
      });
    }
  }

  const geo = new THREE.IcosahedronGeometry(1.0, 2);
  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: {
      uSunDir: { value: new THREE.Vector3(-0.78, 0.32, -0.52).normalize() },
      uSunCol: { value: new THREE.Color(0xffffff) },
      uShadowCol: { value: new THREE.Color(0x9ab2cc) },
      uFogColor: { value: new THREE.Color(0xc9e0f2) },
      uFogDensity: { value: 0.000052 },
      uOpacity: { value: 0.82 },
    },
    vertexShader: `
      varying vec3 vNormalW;
      varying vec3 vWorldPos;
      varying float vLocalY;
      void main() {
        vLocalY = position.y;
        vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        mat3 instN = mat3(instanceMatrix);
        vNormalW = normalize(mat3(modelMatrix) * instN * normal);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: `
      uniform vec3 uSunDir;
      uniform vec3 uSunCol;
      uniform vec3 uShadowCol;
      uniform vec3 uFogColor;
      uniform float uFogDensity;
      uniform float uOpacity;
      varying vec3 vNormalW;
      varying vec3 vWorldPos;
      varying float vLocalY;
      void main() {
        vec3 N = normalize(vNormalW);
        vec3 V = normalize(cameraPosition - vWorldPos);
        float dist = length(cameraPosition - vWorldPos);

        // Crisp sunlit white cumulus top (#ffffff) + soft cool sky-blue underside (#9ab2cc)
        float ndl = dot(N, normalize(uSunDir)) * 0.5 + 0.5;
        float heightLit = smoothstep(-0.75, 0.65, vLocalY);
        float lit = clamp(ndl * 0.55 + heightLit * 0.55, 0.0, 1.0);
        vec3 col = mix(uShadowCol, uSunCol, lit);

        // Soft cloud rim & aerial-perspective distance blend
        float edgeSoft = pow(max(dot(N, V), 0.0), 1.15);
        float fogFactor = 1.0 - exp(-pow(dist * uFogDensity * 0.65, 1.35));
        col = mix(col, uFogColor, clamp(fogFactor, 0.0, 0.84));
        col = min(col, vec3(0.89)); // crisp white without triggering daylight bloom > 0.92!

        float alpha = clamp(edgeSoft * uOpacity * (1.0 - fogFactor * 0.35), 0.0, 0.88);
        if (alpha < 0.02) discard;
        gl_FragColor = vec4(col, alpha);
      }
    `,
  });

  const mesh = new THREE.InstancedMesh(geo, mat, totalPuffs);
  mesh.frustumCulled = false;
  return { mesh, mat, seeds };
}

export function createAtmosphereRig(): AtmosphereRig {
  const group = new THREE.Group();
  group.name = "openxvs_physical_atmosphere";

  const sky = createPhysicalSkyDome();
  group.add(sky.mesh);

  const clouds = createVolumetricCloudDeck();
  group.add(clouds.mesh);

  const dummy = new THREE.Object3D();
  const scratchSunTop = new THREE.Color();
  const scratchShadeBot = new THREE.Color();
  const lightingScratch: MutableAtmosphereLightingState = {
    sunDir: new THREE.Vector3(),
    sunElevationDeg: 0,
    daylightFactor: 0,
    goldenFactor: 0,
    sunColor: new THREE.Color(),
    sunIntensity: 0,
    hemiSkyColor: new THREE.Color(),
    hemiGroundColor: new THREE.Color(),
    hemiIntensity: 0,
    fogColor: new THREE.Color(),
  };

  let driftX = 0.0;
  let driftZ = 0.0;
  let cloudsInitialized = false;
  let lastCloudDriftX = 0.0;
  let lastCloudDriftZ = 0.0;
  let lastCloudCover = -1.0;

  return {
    group,
    skyMesh: sky.mesh,
    cloudsMesh: clouds.mesh,
    computeLighting(timeOfDayH: number, mode: VisualWorldMode, cloudCover: number): AtmosphereLightingState {
      return computeSolarAtmosphere(timeOfDayH, mode, cloudCover);
    },
    update(
      simTimeS: number,
      dtS: number,
      cameraPos: THREE.Vector3,
      weather: WeatherState,
      mode: VisualWorldMode,
    ): AtmosphereLightingState {
      const lit = computeSolarAtmosphereInto(
        weather.timeOfDayH,
        mode,
        weather.cloudCover,
        lightingScratch,
      );
      const isObsidian = mode === "obsidian";

      sky.mesh.position.copy(cameraPos);
      sky.mat.uniforms.uSunDir.value.copy(lit.sunDir);
      sky.mat.uniforms.uDaylight.value = lit.daylightFactor;
      sky.mat.uniforms.uGolden.value = lit.goldenFactor;
      sky.mat.uniforms.uObsidian.value = isObsidian ? 1.0 : 0.0;
      sky.mat.uniforms.uCloudCover.value = weather.cloudCover;
      sky.mat.uniforms.uTime.value = simTimeS;

      const windMps = weather.crosswindKt * 0.514444;
      driftX += windMps * dtS * 1.4;
      driftZ += windMps * dtS * 0.65;

      const showClouds = weather.cloudCover > 0.02 && (!isObsidian || weather.cloudCover > 0.35);
      clouds.mesh.visible = showClouds;
      if (showClouds) {
        // Pure crisp sunlit white tops (#ffffff) + cool sky-blue undersides (#9ab2cc)
        if (isObsidian) {
          scratchSunTop.setHex(0x1f3554);
          scratchShadeBot.setHex(0x07101f);
        } else {
          scratchSunTop
            .copy(COL_CLOUD_NIGHT_TOP)
            .lerp(COL_CLOUD_DAY_TOP, lit.daylightFactor)
            .lerp(COL_CLOUD_GOLD_TOP, lit.goldenFactor * 0.25);
          scratchShadeBot
            .copy(COL_CLOUD_NIGHT_BOT)
            .lerp(COL_CLOUD_DAY_BOT, lit.daylightFactor);
        }

        clouds.mat.uniforms.uSunDir.value.copy(lit.sunDir);
        clouds.mat.uniforms.uSunCol.value.copy(scratchSunTop);
        clouds.mat.uniforms.uShadowCol.value.copy(scratchShadeBot);
        clouds.mat.uniforms.uFogColor.value.copy(lit.fogColor);
        clouds.mat.uniforms.uFogDensity.value = clamp(
          0.95 / Math.max(500, weather.visibilityKm * 1000),
          0.000035,
          0.0012,
        );
        clouds.mat.uniforms.uOpacity.value = isObsidian
          ? 0.24 * weather.cloudCover
          : 0.56 + 0.36 * weather.cloudCover;

        // Throttle 342-instance matrix buffer upload (~22 KB) until drift or cover changes meaningfully
        const needsCloudUpload =
          !cloudsInitialized ||
          Math.abs(driftX - lastCloudDriftX) >= 3.0 ||
          Math.abs(driftZ - lastCloudDriftZ) >= 3.0 ||
          Math.abs(weather.cloudCover - lastCloudCover) >= 0.005;

        if (needsCloudUpload) {
          cloudsInitialized = true;
          lastCloudDriftX = driftX;
          lastCloudDriftZ = driftZ;
          lastCloudCover = weather.cloudCover;

          const activeLimit = clamp(weather.cloudCover * 1.05, 0.0, 1.0);
          const wrapSpan = 30000.0;
          const coverScale = 0.78 + 0.36 * weather.cloudCover;
          let visibleCount = 0;
          for (let i = 0; i < clouds.seeds.length; i++) {
            const s = clouds.seeds[i];
            if (s.densityThreshold > activeLimit) continue;
            const wx = (((s.x + driftX + wrapSpan * 0.5) % wrapSpan) + wrapSpan) % wrapSpan - wrapSpan * 0.5;
            const wz = (((s.z + driftZ + wrapSpan * 0.5) % wrapSpan) + wrapSpan) % wrapSpan - wrapSpan * 0.5;
            dummy.position.set(wx, s.y, wz);
            dummy.rotation.set(0, s.rotY, 0);
            dummy.scale.set(s.sx * coverScale, s.sy * coverScale, s.sz * coverScale);
            dummy.updateMatrix();
            clouds.mesh.setMatrixAt(visibleCount++, dummy.matrix);
          }
          clouds.mesh.count = visibleCount;
          clouds.mesh.instanceMatrix.needsUpdate = true;
        }
      }

      return lit;
    },
  };
}
