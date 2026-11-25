// Pass 3: Split Solid 3D Aircraft + PAPI/ALSF-2 + Translucent 3D Safety Manifolds
// + Vector Wireframes/HITS/MeshDAA + Soft Radial Gaussian 3DGS Splat Billboards.
// Pure f32 WGSL -> Naga 30.0.2 GLSL ES 3.00 dual-backend parity (WebGPU + WebGL2).

struct Pass3Uniforms {
  viewProj: mat4x4<f32>,
  params: vec4<f32>,     // x = simTimeS, y = efvsEnabled, z = fogContrast, w = globalAlpha
  modeParams: vec4<f32>, // x = billboardMode (0=solid/line/manifold, 1=Gaussian splat, 2=emissive halo), y = gaussianSigma, z = discardThresh, w = 0
};

@group(0) @binding(0) var<uniform> u: Pass3Uniforms;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) dashMeta: vec2<f32>, // mode=0: (distAlongM, dashPeriodM); mode>0: billboard UV in [-1, 1]^2
};

@vertex
fn vs_main(
  @location(0) posRte: vec3<f32>,
  @location(1) color: vec4<f32>,
  @location(2) dashMeta: vec2<f32>
) -> VsOut {
  var out: VsOut;
  out.pos = u.viewProj * vec4<f32>(posRte, 1.0);
  out.color = color;
  out.dashMeta = dashMeta;
  return out;
}

@fragment
fn fs_main(inp: VsOut) -> @location(0) vec4<f32> {
  var rgb = inp.color.rgb;
  var alpha = inp.color.a;

  if (u.modeParams.x > 0.5) {
    // Billboard UV in [-1, 1]^2 via dashMeta.xy: soft radial Gaussian falloff (zero sharp edges)
    let uv = inp.dashMeta;
    let r2 = dot(uv, uv);
    let sigma = max(u.modeParams.y, 0.5);
    let g = exp(-sigma * r2);
    if (g < u.modeParams.z) {
      discard;
    }
    if (u.modeParams.x > 1.5) {
      // Emissive radial-halo mode (PAPI lamps, ALSF-2 strobes, aircraft nav/anti-collision beacons)
      let core = exp(-11.0 * r2);
      rgb = clamp(rgb + vec3<f32>(0.55, 0.55, 0.55) * core, vec3<f32>(0.0), vec3<f32>(1.0));
      alpha = clamp(alpha * g, 0.0, 1.0);
    } else {
      // Soft radial 3DGS Gaussian splat mode (Pass 2b SPZ)
      alpha = clamp(alpha * g, 0.0, 1.0);
    }
  } else {
    // dashMeta.y > 0 => dashed line; dashMeta.y < -0.5 => pulsing alert strobe / manifold highlight
    if (inp.dashMeta.y > 0.001) {
      let phase = fract((inp.dashMeta.x + u.params.x * 15.0) / inp.dashMeta.y);
      if (phase > 0.55) {
        discard;
      }
    }
    if (inp.dashMeta.y < -0.5) {
      let strobe = 0.45 + 0.55 * pow(max(sin(u.params.x * 9.0 + inp.dashMeta.x), 0.0), 4.0);
      rgb = clamp(rgb * (0.85 + 0.65 * strobe), vec3<f32>(0.0), vec3<f32>(1.0));
      alpha = clamp(alpha * (0.75 + 0.25 * strobe), 0.0, 1.0);
    }
  }

  // In 8-14 um LWIR EFVS mode, keep runway markings, hot aircraft engines & safety geometry white-hot/phosphor luminous
  if (u.params.y > 0.5) {
    let luma = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
    let thermalBoost = mix(vec3<f32>(0.28, 0.95, 0.58), vec3<f32>(0.96, 1.0, 0.95), clamp(luma * 1.25, 0.0, 1.0));
    rgb = mix(rgb, thermalBoost, 0.42);
  }
  return vec4<f32>(rgb, clamp(alpha * u.params.w, 0.0, 1.0));
}
