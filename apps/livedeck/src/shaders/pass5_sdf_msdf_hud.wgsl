// Pass 5: Single-Pass Analytical SDF Vector HUD + B612 Mono MSDF Typography.
// Zero Canvas2D texture uploads; exact WebGPU + Naga WebGL2 pixel-center alignment.

struct Pass5Uniforms {
  viewport: vec4<f32>, // x=width, y=height, z=atlasWidth, w=atlasHeight
  hudParams: vec4<f32>, // x=msdfPxRange (4.0), y=simTimeS, z=efvsEnabled, w=simplexMode
};

@group(0) @binding(0) var<uniform> u: Pass5Uniforms;
@group(0) @binding(1) var texMsdf: texture_2d<f32>;
@group(0) @binding(2) var sampMsdf: sampler;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) primMeta: vec4<f32>, // x=kind(0=SDF box/line,1=SDF ring,2=MSDF glyph,3=pill), y=halfW, z=halfH/stroke, w=radius
};

@vertex
fn vs_main(
  @location(0) posPx: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) color: vec4<f32>,
  @location(3) primMeta: vec4<f32>
) -> VsOut {
  let w = max(u.viewport.x, 1.0);
  let h = max(u.viewport.y, 1.0);
  // Map screen pixel coordinate (0..W, 0..H from top-left) directly to WebGPU NDC
  let xNdc = (2.0 * posPx.x) / w - 1.0;
  let yNdc = 1.0 - (2.0 * posPx.y) / h;

  var out: VsOut;
  out.pos = vec4<f32>(xNdc, yNdc, 0.01, 1.0);
  out.uv = uv;
  out.color = color;
  out.primMeta = primMeta;
  return out;
}

fn median3(a: f32, b: f32, c: f32) -> f32 {
  return max(min(a, b), min(max(a, b), c));
}

fn sdRoundedBox(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
  let q = abs(p) - b + vec2<f32>(r, r);
  return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(inp: VsOut) -> @location(0) vec4<f32> {
  let kind = inp.primMeta.x;
  var alpha: f32 = inp.color.a;

  if (kind < 0.5) {
    // Kind 0: Analytical SDF line segment / crisp rounded box (uv in [-1, 1])
    let halfExt = vec2<f32>(inp.primMeta.y, inp.primMeta.z);
    let pLocal = inp.uv * halfExt;
    let dist = sdRoundedBox(pLocal, max(halfExt - vec2<f32>(0.75, 0.75), vec2<f32>(0.25, 0.25)), inp.primMeta.w);
    let cov = 1.0 - smoothstep(-0.6, 0.6, dist);
    alpha = alpha * cov;
  } else if (kind < 1.5) {
    // Kind 1: Analytical SDF circle / reticle ring (primMeta.y = radiusPx, primMeta.z = strokeHalfPx)
    let radius = inp.primMeta.y;
    let strokeHalf = max(inp.primMeta.z, 0.6);
    let pLocal = inp.uv * (radius + strokeHalf + 1.5);
    let dist = abs(length(pLocal) - radius) - strokeHalf;
    let cov = 1.0 - smoothstep(-0.65, 0.65, dist);
    alpha = alpha * cov;
  } else if (kind < 2.5) {
    // Kind 2: B612 Mono MSDF glyph (textureSampleLevel is valid in non-uniform control flow)
    let sampleRgb = textureSampleLevel(texMsdf, sampMsdf, inp.uv, 0.0).rgb;
    let sd = median3(sampleRgb.r, sampleRgb.g, sampleRgb.b) - 0.5;
    let pxRange = max(u.hudParams.x, 1.0);
    let screenPxDist = sd * pxRange * max(inp.primMeta.y, 0.75);
    let cov = clamp(screenPxDist + 0.5, 0.0, 1.0);
    alpha = alpha * cov;
  } else {
    // Kind 3: Rounded HUD pill badge (filled with subtle crisp outline)
    let halfExt = vec2<f32>(inp.primMeta.y, inp.primMeta.z);
    let pLocal = inp.uv * halfExt;
    let dist = sdRoundedBox(pLocal, max(halfExt - vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 1.0)), inp.primMeta.w);
    let cov = 1.0 - smoothstep(-0.7, 0.7, dist);
    alpha = alpha * cov;
  }

  if (alpha < 0.01) {
    discard;
  }
  return vec4<f32>(inp.color.rgb, alpha);
}
