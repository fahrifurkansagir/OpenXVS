// Pass 1: 6-Camera Spherical Distributed-Aperture System (XVS6-CUBE-v1) & 8-14 um LWIR EFVS
// Strictly obeys CONVENTIONS.md v1.0 (NED, BODY FRD, OpenCV CAM, camera order F,R,B,L,U,D).

struct Pass1Uniforms {
  rBodyCam0: vec4<f32>,    // xyz = R_body_cam col 0, w = fx
  rBodyCam1: vec4<f32>,    // xyz = R_body_cam col 1, w = fy
  rBodyCam2: vec4<f32>,    // xyz = R_body_cam col 2, w = cx
  rNedBody0: vec4<f32>,    // xyz = R_ned_body col 0, w = cy
  rNedBody1: vec4<f32>,    // xyz = R_ned_body col 1, w = viewportWidth
  rNedBody2: vec4<f32>,    // xyz = R_ned_body col 2, w = viewportHeight
  sunDirNed: vec4<f32>,    // xyz = unit sun vector in NED, w = horizonDipRad
  envParams: vec4<f32>,    // x = fogContrast (0.22..1.0), y = efvsEnabled (0/1), z = camRBiasRad, w = simTimeS
  healthParams: vec4<f32>, // x = camHealthMask (0..63), y = glareStrength, z = altitudeM, w = simplexMode
};

@group(0) @binding(0) var<uniform> u: Pass1Uniforms;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@location(0) posNdc: vec2<f32>) -> VsOut {
  var out: VsOut;
  out.pos = vec4<f32>(posNdc, 0.9999, 1.0);
  out.ndc = posNdc;
  return out;
}

fn bitEnabled(maskF: f32, bitIdx: i32) -> bool {
  let m = u32(max(maskF + 0.5, 0.0));
  return ((m >> u32(bitIdx)) & 1u) != 0u;
}

@fragment
fn fs_main(inp: VsOut) -> @location(0) vec4<f32> {
  let fx = u.rBodyCam0.w;
  let fy = u.rBodyCam1.w;
  let cx = u.rBodyCam2.w;
  let cy = u.rNedBody0.w;

  // CONVENTIONS.md §5: integer pixel centers (u, v); fragment center is (u + 0.5, v + 0.5)
  let px = inp.pos.x - 0.5;
  let py = inp.pos.y - 0.5;

  let rayCam = normalize(vec3<f32>((px - cx) / fx, (py - cy) / fy, 1.0));

  let rBodyCam = mat3x3<f32>(u.rBodyCam0.xyz, u.rBodyCam1.xyz, u.rBodyCam2.xyz);
  let rNedBody = mat3x3<f32>(u.rNedBody0.xyz, u.rNedBody1.xyz, u.rNedBody2.xyz);

  var rayBody = normalize(rBodyCam * rayCam);

  // Determine active XVS6-CUBE-v1 camera sector in F(0), R(1), B(2), L(3), U(4), D(5) order
  let ax = abs(rayBody.x);
  let ay = abs(rayBody.y);
  let az = abs(rayBody.z);
  var camIdx: i32 = 0;
  var seamMetric: f32 = 1.0;
  if (ax >= ay && ax >= az) {
    camIdx = select(2, 0, rayBody.x >= 0.0);
    seamMetric = ax - max(ay, az);
  } else if (ay >= ax && ay >= az) {
    camIdx = select(3, 1, rayBody.y >= 0.0);
    seamMetric = ay - max(ax, az);
  } else {
    camIdx = select(4, 5, rayBody.z >= 0.0);
    seamMetric = az - max(ax, ay);
  }

  let healthy = bitEnabled(u.healthParams.x, camIdx);

  // Apply physical Cam-R extrinsic misalignment perturbation in R sector (M1 mission)
  let camRBias = u.envParams.z;
  if (camIdx == 1 && abs(camRBias) > 0.00001) {
    let cb = cos(camRBias);
    let sb = sin(camRBias);
    // Rotation around BODY forward axis (+x) tilts Cam-R horizon visibly
    rayBody = normalize(vec3<f32>(
      rayBody.x,
      rayBody.y * cb - rayBody.z * sb,
      rayBody.y * sb + rayBody.z * cb
    ));
  }

  let rayNed = normalize(rNedBody * rayBody);
  // NED: +z is Down, so elevation above local horizontal is -asin(rayNed.z)
  let elev = -asin(clamp(rayNed.z, -1.0, 1.0));
  let horizonDip = u.sunDirNed.w;
  let elevAboveLimb = elev + horizonDip;

  let fogContrast = clamp(u.envParams.x, 0.08, 1.0);
  let efvs = u.envParams.y;

  // Physical Rayleigh + Mie atmospheric scattering sky / Earth limb model
  let sunDir = normalize(u.sunDirNed.xyz);
  let cosTheta = clamp(dot(rayNed, sunDir), -1.0, 1.0);
  let glare = u.healthParams.y;
  let sunDisc = smoothstep(0.99935, 0.99988, cosTheta) * 1.65 * glare;
  let altKm = clamp(u.healthParams.z * 0.001, 0.0, 20.0);
  let atmDensity = exp(-altKm / 8.4);

  // Shared Rayleigh horizon haze tint (matches Pass 2 aerial perspective seamlessly)
  let rayleighHorizon = vec3<f32>(0.68, 0.77, 0.86);

  var color: vec3<f32>;
  if (elevAboveLimb >= 0.0) {
    // Kasten-Young optical air-mass approximation above WGS-84 limb
    let elDeg = elevAboveLimb * 57.2957795;
    let airMass = min(1.0 / (sin(elevAboveLimb) + 0.15 * pow(elDeg + 3.885, -1.253)), 32.0);
    let sunElDeg = max(-asin(clamp(sunDir.z, -1.0, 1.0)) * 57.2957795, 1.5);
    let sunAirMass = min(1.0 / (sin(sunElDeg * 0.01745329) + 0.15 * pow(sunElDeg + 3.885, -1.253)), 28.0);

    // Wavelength-dependent Rayleigh coefficients (680 nm / 550 nm / 440 nm)
    let betaR = vec3<f32>(0.058, 0.135, 0.331);
    let tauR = betaR * airMass * atmDensity * 0.38;
    let sunTrans = exp(-betaR * sunAirMass * 0.22);
    let phaseR = 0.75 * (1.0 + cosTheta * cosTheta);

    let rayleighScatter = (vec3<f32>(1.0) - exp(-tauR)) * phaseR * sunTrans * vec3<f32>(0.90, 0.96, 1.12);
    let zenithBlue = vec3<f32>(0.07, 0.21, 0.46) * mix(0.62, 1.0, atmDensity);
    let horizonBand = exp(-elevAboveLimb * 7.5);
    color = zenithBlue + rayleighScatter * 0.62;
    color = mix(color, rayleighHorizon, horizonBand * 0.78);

    // Henyey-Greenstein Mie forward-scattering solar aureole (g = 0.76)
    let g = 0.76;
    let g2 = g * g;
    let hgDenom = max(1.0 + g2 - 2.0 * g * cosTheta, 0.015);
    let phaseM = (1.0 - g2) / (12.5663706 * pow(hgDenom, 1.5));
    let mieOptical = 1.0 - exp(-0.085 * airMass * atmDensity);
    let mieAureole = min(phaseM * 0.38 * mieOptical, 1.6) * glare;
    let warmHorizonGlow = pow(max(cosTheta, 0.0), 3.0) * horizonBand * 0.18 * glare;
    color = color + vec3<f32>(1.0, 0.91, 0.76) * (mieAureole + warmHorizonGlow + sunDisc);

    // Subtle high-altitude cirrus bands with slow atmospheric drift
    let azim = atan2(rayNed.y, rayNed.x);
    let drift = u.envParams.w * 0.004;
    let c1 = sin(azim * 7.0 + elevAboveLimb * 32.0 + drift) * cos(azim * 13.0 - elevAboveLimb * 18.0 - drift * 0.7);
    let c2 = sin(azim * 19.0 - elevAboveLimb * 48.0 + drift * 1.3) * 0.5;
    let cirrusVal = smoothstep(0.18, 0.86, (c1 + c2) * 0.42 + 0.48);
    let cloudMask = smoothstep(0.05, 0.32, elevAboveLimb) * smoothstep(0.78, 0.24, elevAboveLimb);
    let cirrusCol = mix(vec3<f32>(0.86, 0.90, 0.95), vec3<f32>(1.0, 0.94, 0.86), pow(max(cosTheta, 0.0), 4.0) * 0.6);
    color = mix(color, cirrusCol, cirrusVal * cloudMask * 0.16);

    // Gentle shoulder highlight compression so solar disc & Mie aureole stay smooth
    color = color / (vec3<f32>(1.0) + max(color - vec3<f32>(0.88), vec3<f32>(0.0)) * 0.55);
  } else {
    // Below WGS-84 limb: SF Bay dark maritime tone fading seamlessly to Rayleigh horizon haze
    let dep = clamp(-elevAboveLimb * 4.5, 0.0, 1.0);
    let bayCol = vec3<f32>(0.045, 0.125, 0.205);
    color = mix(rayleighHorizon * 0.88, bayCol, pow(dep, 0.58));
    // Subtle distant water sun-path reflection beyond 40 km terrain ring
    let glintMask = pow(max(cosTheta, 0.0), 18.0) * exp(-dep * 2.2) * 0.26 * glare;
    color = color + vec3<f32>(1.0, 0.92, 0.76) * glintMask;
  }

  // Fog / Polar-haze contrast degradation (e.g., 0.22x in M4 KSFO fog)
  let fogDensity = 1.0 - fogContrast;
  let fogBand = exp(-abs(elevAboveLimb) * 10.0);
  let fogColor = vec3<f32>(0.64, 0.69, 0.73);
  color = mix(color, fogColor, clamp(fogDensity * (0.68 + 0.32 * fogBand), 0.0, 0.92));

  // Physics-inspired 8-14 um LWIR EFVS mode: cuts sharply through visual fog with aviation phosphor-green / white-hot palette
  if (efvs > 0.5) {
    let skyCold = mix(0.16, 0.02, clamp(pow(max(elevAboveLimb, 0.0) * 2.4, 0.65), 0.0, 1.0));
    let groundWarm = mix(0.28, 0.54, clamp(-elevAboveLimb * 3.5, 0.0, 1.0));
    let limbGlow = exp(-abs(elevAboveLimb) * 42.0) * 0.22;
    let limbContrast = select(groundWarm, skyCold, elevAboveLimb >= 0.0);
    // Subtle microbolometer row-noise / thermal scanline texture
    let scan = 0.96 + 0.04 * sin(py * 1.5 + u.envParams.w * 8.0);
    let thermalLuma = clamp((limbContrast + limbGlow + sunDisc * 0.75) * scan, 0.02, 0.98);
    // Crisp aviation HUD phosphor-green (cool) -> bright white-hot (hot) ramp
    let coolCol = vec3<f32>(0.02, 0.14, 0.09);
    let midCol = vec3<f32>(0.18, 0.86, 0.48);
    let hotCol = vec3<f32>(0.92, 1.0, 0.94);
    if (thermalLuma < 0.55) {
      color = mix(coolCol, midCol, thermalLuma / 0.55);
    } else {
      color = mix(midCol, hotCol, (thermalLuma - 0.55) / 0.45);
    }
  }

  // Subtle DAS cube-face seam line
  let seamLine = 1.0 - smoothstep(0.0, 0.008, seamMetric);
  color = mix(color, vec3<f32>(0.22, 0.48, 0.52), seamLine * 0.22);

  // Isolated / faulted camera mask (cross-hatch red-amber occlusion)
  if (!healthy) {
    let hatch = step(0.5, fract((px + py) * 0.06));
    let maskCol = mix(vec3<f32>(0.14, 0.04, 0.04), vec3<f32>(0.42, 0.09, 0.06), hatch);
    color = mix(color * 0.18, maskCol, 0.86);
  }

  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
