// Pass 2: Dual-Ring 16-bit USGS 3DEP (GEOID18 -> WGS-84) CDLOD Terrain Clipmap
// with Stage-1 3-Tier Real USGS/USDA Satellite Orthophoto Draping (2x2 km Core2K @ ~0.98 m/texel,
// 10x10 km Inner, 40x40 km Outer), Analytic Central-Difference Normals, Distance-Faded Micro-Detail,
// Animated Fresnel/GGX San Francisco Bay Water, TSO-C151d TAWS FLTA Shading, and 8-14 um LWIR EFVS.

struct Pass2Uniforms {
  viewProj: mat4x4<f32>,
  originRteEcef: vec4<f32>, // xyz = p_origin_ecef - p_cam_ecef (f64 RTE), w = hMinM (-100.0)
  rEcefNed0: vec4<f32>,     // xyz = col 0 (North) of R_ecef_ned at KSFO origin, w = hStepM (0.1)
  rEcefNed1: vec4<f32>,     // xyz = col 1 (East), w = ownAltWgs84M
  rEcefNed2: vec4<f32>,     // xyz = col 2 (Down), w = fogContrast
  renderParams: vec4<f32>,  // x = efvsEnabled (0/1), y = proceduralCiMode (0/1), z = simTimeS, w = ringTexSize (256.0)
  orthoParams: vec4<f32>,   // x = orthoEnabled (0/1), y = orthoStrength (0..1), z = innerExtentM (10000), w = outerExtentM (40000)
  sunDirNed: vec4<f32>,     // xyz = unit sun vector in NED (matches Pass 1), w = coreOrthoEnabled (0/1)
  coreBoundsNe: vec4<f32>,  // x = coreCenterN, y = coreCenterE, z = coreHalfExtentM (1000), w = 0
  camPosNeDown: vec4<f32>,  // x = camNorthM, y = camEastM, z = camDownM (KSFO origin NED), w = 0
};

@group(0) @binding(0) var<uniform> u: Pass2Uniforms;
@group(0) @binding(1) var texInner: texture_2d<f32>;
@group(0) @binding(2) var texOuter: texture_2d<f32>;
@group(0) @binding(3) var texOrthoInner: texture_2d<f32>;
@group(0) @binding(4) var texOrthoOuter: texture_2d<f32>;
@group(0) @binding(5) var orthoSampler: sampler;
@group(0) @binding(6) var texOrthoCore: texture_2d<f32>;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) hWgs84: f32,
  @location(1) clearanceM: f32,
  @location(2) distM: f32,
  @location(3) localNe: vec2<f32>,
  @location(4) ringId: f32,
  @location(5) uv: vec2<f32>,
};

fn decodeHeight16(texel: vec4<f32>, hMin: f32, hStep: f32) -> f32 {
  let hi = floor(texel.r * 255.0 + 0.5);
  let lo = floor(texel.g * 255.0 + 0.5);
  let q = hi * 256.0 + lo;
  return hMin + q * hStep;
}

fn sampleDemClamped(isInner: bool, ix: i32, iy: i32, maxIdx: i32, hMin: f32, hStep: f32) -> f32 {
  let cx = clamp(ix, 0, maxIdx);
  let cy = clamp(iy, 0, maxIdx);
  let tInner = textureLoad(texInner, vec2<i32>(cx, cy), 0);
  let tOuter = textureLoad(texOuter, vec2<i32>(cx, cy), 0);
  let texel = select(tOuter, tInner, isInner);
  return decodeHeight16(texel, hMin, hStep);
}

@vertex
fn vs_main(
  @location(0) neCoord: vec2<f32>,   // (northM, eastM) relative to KSFO origin
  @location(1) uvRing: vec3<f32>     // (u, v, ringFlag: 0=inner, 1=outer)
) -> VsOut {
  let hMin = u.originRteEcef.w;
  let hStep = u.rEcefNed0.w;
  let texSize = max(u.renderParams.w, 2.0);
  let ix = i32(clamp(floor(uvRing.x * (texSize - 1.0) + 0.5), 0.0, texSize - 1.0));
  let iy = i32(clamp(floor(uvRing.y * (texSize - 1.0) + 0.5), 0.0, texSize - 1.0));

  var sampleH: vec4<f32>;
  if (uvRing.z < 0.5) {
    sampleH = textureLoad(texInner, vec2<i32>(ix, iy), 0);
  } else {
    sampleH = textureLoad(texOuter, vec2<i32>(ix, iy), 0);
  }
  let hWgs84 = decodeHeight16(sampleH, hMin, hStep);

  // Earth curvature sag over local NED tangent plane so 40x40 km terrain hugs WGS-84 ellipsoid
  let r2 = neCoord.x * neCoord.x + neCoord.y * neCoord.y;
  let curvDropM = r2 / (2.0 * 6378137.0);
  let downM = -hWgs84 + curvDropM;

  let posRte =
    u.originRteEcef.xyz +
    u.rEcefNed0.xyz * neCoord.x +
    u.rEcefNed1.xyz * neCoord.y +
    u.rEcefNed2.xyz * downM;

  var out: VsOut;
  out.pos = u.viewProj * vec4<f32>(posRte, 1.0);
  out.hWgs84 = hWgs84;
  out.clearanceM = u.rEcefNed1.w - hWgs84;
  out.distM = length(posRte);
  out.localNe = neCoord;
  out.ringId = uvRing.z;
  out.uv = uvRing.xy;
  return out;
}

@fragment
fn fs_main(inp: VsOut) -> @location(0) vec4<f32> {
  // Hypsometric terrain height relative to KSFO sea/bay level (WGS-84 h ~= -32.63 m)
  let hMslApprox = inp.hWgs84 + 32.63;
  let uvClamped = clamp(inp.uv, vec2<f32>(0.0), vec2<f32>(1.0));
  let isInner = inp.ringId < 0.5;

  // 3-tier hardware-bilinear satellite orthophoto sampling (uniform control flow):
  //   - texOrthoOuter (40x40 km), texOrthoInner (10x10 km), texOrthoCore (2x2 km @ ~0.98 m/texel)
  let orthoInnerRgb = textureSampleLevel(texOrthoInner, orthoSampler, uvClamped, 0.0).rgb;
  let orthoOuterRgb = textureSampleLevel(texOrthoOuter, orthoSampler, uvClamped, 0.0).rgb;

  let coreHalf = max(u.coreBoundsNe.z, 1.0);
  let coreSpan = coreHalf * 2.0;
  let coreU = (inp.localNe.y - (u.coreBoundsNe.y - coreHalf)) / coreSpan;
  let coreV = ((u.coreBoundsNe.x + coreHalf) - inp.localNe.x) / coreSpan;
  let coreUv = clamp(vec2<f32>(coreU, coreV), vec2<f32>(0.0), vec2<f32>(1.0));
  let orthoCoreRgb = textureSampleLevel(texOrthoCore, orthoSampler, coreUv, 0.0).rgb;

  let coreEdge = min(min(coreU, 1.0 - coreU), min(coreV, 1.0 - coreV));
  let coreMask = smoothstep(0.0, 0.08, coreEdge) * clamp(u.sunDirNed.w, 0.0, 1.0);
  let baseOrthoRgb = select(orthoOuterRgb, orthoInnerRgb, isInner);
  let orthoRgb = mix(baseOrthoRgb, orthoCoreRgb, coreMask);

  let useOrtho = (u.orthoParams.x > 0.5) && (u.renderParams.y < 0.5);
  let orthoBlend = select(0.0, clamp(u.orthoParams.y, 0.0, 1.0), useOrtho);

  // Analytic C1-smooth central-difference terrain normal in KSFO NED from 16-bit 3DEP DEM
  let hMin = u.originRteEcef.w;
  let hStep = u.rEcefNed0.w;
  let texSize = max(u.renderParams.w, 2.0);
  let maxIdx = i32(texSize - 1.0);
  let fx = uvClamped.x * (texSize - 1.0);
  let fy = uvClamped.y * (texSize - 1.0);
  let ix = i32(floor(fx));
  let iy = i32(floor(fy));
  let tx = fract(fx);
  let ty = fract(fy);

  let ringExtentM = select(max(u.orthoParams.w, 1000.0), max(u.orthoParams.z, 1000.0), isInner);
  let cellM = ringExtentM / (texSize - 1.0);

  let hC0 = sampleDemClamped(isInner, ix, iy, maxIdx, hMin, hStep);
  let hR0 = sampleDemClamped(isInner, ix + 1, iy, maxIdx, hMin, hStep);
  let hC1 = sampleDemClamped(isInner, ix, iy + 1, maxIdx, hMin, hStep);
  let hR1 = sampleDemClamped(isInner, ix + 1, iy + 1, maxIdx, hMin, hStep);

  let hL0 = sampleDemClamped(isInner, ix - 1, iy, maxIdx, hMin, hStep);
  let hRR0 = sampleDemClamped(isInner, ix + 2, iy, maxIdx, hMin, hStep);
  let hL1 = sampleDemClamped(isInner, ix - 1, iy + 1, maxIdx, hMin, hStep);
  let hRR1 = sampleDemClamped(isInner, ix + 2, iy + 1, maxIdx, hMin, hStep);

  let hU0 = sampleDemClamped(isInner, ix, iy - 1, maxIdx, hMin, hStep);
  let hU1 = sampleDemClamped(isInner, ix + 1, iy - 1, maxIdx, hMin, hStep);
  let hD0 = sampleDemClamped(isInner, ix, iy + 2, maxIdx, hMin, hStep);
  let hD1 = sampleDemClamped(isInner, ix + 1, iy + 2, maxIdx, hMin, hStep);

  let dE0 = mix((hR0 - hL0) * 0.5, (hRR0 - hC0) * 0.5, tx);
  let dE1 = mix((hR1 - hL1) * 0.5, (hRR1 - hC1) * 0.5, tx);
  let dH_dEast = mix(dE0, dE1, ty) / cellM;

  // v=0 is North (iy - 1 is North, iy + 1 is South)
  let dN0 = mix((hU0 - hC1) * 0.5, (hC0 - hD0) * 0.5, ty);
  let dN1 = mix((hU1 - hR1) * 0.5, (hR0 - hD1) * 0.5, ty);
  let dH_dNorth = mix(dN0, dN1, tx) / cellM;

  // Upward surface normal in NED (+z is Down, so flat horizontal terrain normal is (0, 0, -1))
  let terrainNormal = normalize(vec3<f32>(-dH_dNorth, -dH_dEast, -1.0));
  let slopeMag = clamp(sqrt(dH_dNorth * dH_dNorth + dH_dEast * dH_dEast) * 1.4, 0.0, 0.55);

  // Sun + hemisphere sky lighting (matching Pass 1 afternoon SF Bay sun)
  let sunDir = normalize(u.sunDirNed.xyz);
  let nDotL = max(dot(terrainNormal, sunDir), 0.0);
  let upFactor = clamp(-terrainNormal.z * 0.5 + 0.5, 0.0, 1.0);
  let hemiAmbient = mix(vec3<f32>(0.28, 0.30, 0.32), vec3<f32>(0.42, 0.49, 0.59), upFactor);
  let sunLight = vec3<f32>(1.05, 0.99, 0.90) * nDotL * 0.86;
  let terrainIrradiance = hemiAmbient * 0.95 + sunLight;

  // Distance-faded procedural micro-detail so close-up tarmac, grass & hills read crisply
  let microFade = 1.0 - smoothstep(140.0, 1800.0, inp.distM);
  let m1 = sin(inp.localNe.x * 1.37 + inp.localNe.y * 0.91) * cos(inp.localNe.x * 0.73 - inp.localNe.y * 1.53);
  let m2 = sin(inp.localNe.x * 4.8 - inp.localNe.y * 3.9) * cos(inp.localNe.x * 3.1 + inp.localNe.y * 5.2);
  let microDetail = 1.0 + (m1 * 0.6 + m2 * 0.4) * 0.055 * microFade;

  // Hypsometric fallback albedo (used in procedural_ci or before Stage-1 ortho hydration)
  var hypsoCol: vec3<f32>;
  if (hMslApprox <= 0.8) {
    hypsoCol = vec3<f32>(0.045, 0.13, 0.21);
  } else if (hMslApprox < 45.0) {
    let t = clamp((hMslApprox - 0.8) / 44.2, 0.0, 1.0);
    hypsoCol = mix(vec3<f32>(0.15, 0.24, 0.19), vec3<f32>(0.24, 0.33, 0.22), t);
  } else if (hMslApprox < 250.0) {
    let t = clamp((hMslApprox - 45.0) / 205.0, 0.0, 1.0);
    hypsoCol = mix(vec3<f32>(0.24, 0.33, 0.22), vec3<f32>(0.39, 0.35, 0.26), t);
  } else {
    let t = clamp((hMslApprox - 250.0) / 350.0, 0.0, 1.0);
    hypsoCol = mix(vec3<f32>(0.39, 0.35, 0.26), vec3<f32>(0.56, 0.50, 0.44), t);
  }

  let landAlbedo = mix(hypsoCol, orthoRgb, orthoBlend) * microDetail;
  let litLand = clamp(landAlbedo * terrainIrradiance, vec3<f32>(0.0), vec3<f32>(1.0));

  // Animated calm-coastal San Francisco Bay water with distance & grazing-angle wave attenuation
  // (eliminates distant sinusoidal moire rings while keeping silky Fresnel sky reflection & GGX sun shimmer)
  let tSim = u.renderParams.z;
  let toCamNed = vec3<f32>(
    u.camPosNeDown.x - inp.localNe.x,
    u.camPosNeDown.y - inp.localNe.y,
    u.camPosNeDown.z + inp.hWgs84
  );
  let viewDir = normalize(toCamNed);
  // In NED, |viewDir.z| is sin(grazing elevation); damp waves smoothly at far range & shallow grazing angles
  let waveFade = smoothstep(2600.0, 120.0, inp.distM) * clamp(abs(viewDir.z) * 2.8, 0.06, 1.0);
  let p1 = inp.localNe.x * 0.035 + inp.localNe.y * 0.048 + tSim * 0.85;
  let p2 = inp.localNe.x * 0.062 - inp.localNe.y * 0.054 - tSim * 1.15;
  let p3 = inp.localNe.x * 0.095 + inp.localNe.y * 0.082 + tSim * 1.45;
  let wN = (cos(p1) * 0.018 + cos(p2) * 0.013 + cos(p3) * 0.007) * waveFade;
  let wE = (cos(p1) * 0.022 - cos(p2) * 0.012 + cos(p3) * 0.006) * waveFade;
  let waterNormal = normalize(vec3<f32>(wN, wE, -1.0));

  let nDotV = max(dot(waterNormal, viewDir), 0.001);
  // Water Schlick Fresnel (F0 = 0.02)
  let fresnel = 0.02 + 0.98 * pow(1.0 - nDotV, 5.0);

  // Reflected Rayleigh sky gradient matching Pass 1
  let reflDir = reflect(-viewDir, waterNormal);
  let reflElev = clamp(-reflDir.z, 0.0, 1.0);
  let skyRefl = mix(vec3<f32>(0.68, 0.77, 0.86), vec3<f32>(0.10, 0.25, 0.50), pow(reflElev, 0.55));
  let deepBayCol = mix(vec3<f32>(0.025, 0.085, 0.155), orthoRgb * 0.58, orthoBlend * 0.16);
  var waterCol = mix(deepBayCol, skyRefl, clamp(fresnel * 0.88 + 0.06, 0.0, 0.94));

  // Subtle GGX microfacet sun shimmer on calm bay ripples (damped smoothly with distance/grazing)
  let halfVec = normalize(sunDir + viewDir);
  let nDotH = max(dot(waterNormal, halfVec), 0.0);
  let alpha = 0.11;
  let a2 = alpha * alpha;
  let ggxDenom = nDotH * nDotH * (a2 - 1.0) + 1.0;
  let ndf = a2 / (3.14159265 * ggxDenom * ggxDenom + 0.0001);
  let waterSunVis = max(dot(waterNormal, sunDir), 0.0);
  let specGlint = min(ndf * fresnel * waterSunVis * (0.28 + 0.26 * waveFade), 1.6);
  waterCol = waterCol + vec3<f32>(1.0, 0.92, 0.78) * specGlint;

  // Smooth shoreline transition from SF Bay water into reclaimed airfield seawall & land
  let landMask = smoothstep(0.15, 1.45, hMslApprox);
  var col = mix(waterCol, litLand, landMask);

  // Subtle 50 m elevation contour + 1 km synthetic vision grid (lighter when real ortho is active)
  let contour = 1.0 - smoothstep(0.0, 1.8, abs(fract(hMslApprox / 50.0 + 0.5) - 0.5) * 50.0);
  let gx = abs(fract(inp.localNe.x / 1000.0 + 0.5) - 0.5) * 1000.0;
  let gy = abs(fract(inp.localNe.y / 1000.0 + 0.5) - 0.5) * 1000.0;
  let gridLine = 1.0 - smoothstep(0.0, 16.0, min(gx, gy));
  let overlayStrength = mix(1.0, 0.18, orthoBlend);
  col = mix(col, vec3<f32>(0.20, 0.46, 0.42), (gridLine * 0.10 + contour * 0.12) * overlayStrength);

  // TSO-C151d TAWS FLTA caution/warning ONLY on genuine close-in CFIT proximity (< 2.2 km);
  // distant coastal hills (> 2.2 km, e.g. San Bruno Mtn) preserve crisp natural California USGS ortho!
  if (hMslApprox > 12.0 && inp.distM < 2200.0) {
    let pulse = 0.65 + 0.35 * sin(u.renderParams.z * 6.0);
    if (inp.clearanceM < 60.0) {
      col = mix(col, vec3<f32>(0.92, 0.14, 0.12), 0.72 * pulse);
    } else if (inp.clearanceM < 150.0) {
      col = mix(col, vec3<f32>(0.95, 0.66, 0.10), 0.48);
    }
  }

  // Atmospheric aerial perspective matching Pass 1 Rayleigh horizon haze + fog contrast degradation
  let fogContrast = clamp(u.rEcefNed2.w, 0.08, 1.0);
  let fogDensity = 1.0 - fogContrast;
  let distFactor = 1.0 - exp(-inp.distM * (0.000028 + fogDensity * 0.00055));
  let fogCol = mix(vec3<f32>(0.68, 0.77, 0.86), vec3<f32>(0.64, 0.69, 0.73), fogDensity);

  if (u.renderParams.x > 0.5) {
    // Dramatic 8-14 um LWIR EFVS: penetrates visual fog with analytic thermal ridge relief + ortho tarmac/coast contrast
    let orthoLuma = dot(orthoRgb, vec3<f32>(0.299, 0.587, 0.114));
    let isBay = hMslApprox <= 0.8;
    let orthoThermalMod = select(0.0, (orthoLuma - 0.32) * 0.42, useOrtho);
    let baseThermal = select(
      clamp(0.42 + clamp(hMslApprox / 520.0, 0.0, 0.34) + slopeMag + orthoThermalMod, 0.28, 0.98),
      clamp(0.11 + 0.04 * contour + select(0.0, orthoLuma * 0.10, useOrtho), 0.06, 0.26),
      isBay
    );
    let lwirAtm = 1.0 - exp(-inp.distM * 0.000032);
    let thermalVal = mix(baseThermal, 0.24, lwirAtm * 0.38);
    let coolCol = vec3<f32>(0.02, 0.13, 0.09);
    let midCol = vec3<f32>(0.20, 0.90, 0.50);
    let hotCol = vec3<f32>(0.93, 1.0, 0.94);
    var lwirCol = select(
      mix(midCol, hotCol, clamp((thermalVal - 0.58) / 0.42, 0.0, 1.0)),
      mix(coolCol, midCol, clamp(thermalVal / 0.58, 0.0, 1.0)),
      thermalVal < 0.58
    );
    // Crisp bright thermal ridge contour lines & synthetic vision range grid
    lwirCol = lwirCol + vec3<f32>(0.18, 0.38, 0.24) * (contour * 0.30 + gridLine * 0.14);
    // Preserve close-in TAWS caution/warning overlays visibly even inside LWIR mode
    if (hMslApprox > 12.0 && inp.distM < 2200.0) {
      let pulse = 0.65 + 0.35 * sin(u.renderParams.z * 6.0);
      if (inp.clearanceM < 60.0) {
        lwirCol = mix(lwirCol, vec3<f32>(1.0, 0.20, 0.18), 0.72 * pulse);
      } else if (inp.clearanceM < 150.0) {
        lwirCol = mix(lwirCol, vec3<f32>(1.0, 0.72, 0.14), 0.50);
      }
    }
    col = lwirCol;
  } else {
    col = mix(col, fogCol, clamp(distFactor, 0.0, 0.94));
  }

  return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
