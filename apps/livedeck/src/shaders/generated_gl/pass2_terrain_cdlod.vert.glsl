#version 300 es

precision highp float;
precision highp int;

struct Pass2Uniforms {
    mat4x4 viewProj;
    vec4 originRteEcef;
    vec4 rEcefNed0_;
    vec4 rEcefNed1_;
    vec4 rEcefNed2_;
    vec4 renderParams;
    vec4 orthoParams;
    vec4 sunDirNed;
    vec4 coreBoundsNe;
    vec4 camPosNeDown;
};
struct VsOut {
    vec4 pos;
    float hWgs84_;
    float clearanceM;
    float distM;
    vec2 localNe;
    float ringId;
    vec2 uv;
};
layout(std140) uniform Pass2Uniforms_block_0Vertex { Pass2Uniforms _group_0_binding_0_vs; };

uniform highp sampler2D _group_0_binding_1_vs;

uniform highp sampler2D _group_0_binding_2_vs;

layout(location = 0) in vec2 _p2vs_location0;
layout(location = 1) in vec3 _p2vs_location1;
smooth out float _vs2fs_location0;
smooth out float _vs2fs_location1;
smooth out float _vs2fs_location2;
smooth out vec2 _vs2fs_location3;
smooth out float _vs2fs_location4;
smooth out vec2 _vs2fs_location5;

float decodeHeight16_(vec4 texel, float hMin, float hStep) {
    float hi = floor(((texel.x * 255.0) + 0.5));
    float lo = floor(((texel.y * 255.0) + 0.5));
    float q = ((hi * 256.0) + lo);
    return (hMin + (q * hStep));
}

float sampleDemClamped(bool isInner, int ix, int iy, int maxIdx, float hMin_1, float hStep_1) {
    int cx = min(max(ix, 0), maxIdx);
    int cy = min(max(iy, 0), maxIdx);
    vec4 tInner = texelFetch(_group_0_binding_1_vs, ivec2(cx, cy), 0);
    vec4 tOuter = texelFetch(_group_0_binding_2_vs, ivec2(cx, cy), 0);
    vec4 texel_1 = (isInner ? tInner : tOuter);
    float _e19 = decodeHeight16_(texel_1, hMin_1, hStep_1);
    return _e19;
}

void main() {
    vec2 neCoord = _p2vs_location0;
    vec3 uvRing = _p2vs_location1;
    vec4 sampleH = vec4(0.0);
    VsOut out_ = VsOut(vec4(0.0), 0.0, 0.0, 0.0, vec2(0.0), 0.0, vec2(0.0));
    float hMin_2 = _group_0_binding_0_vs.originRteEcef.w;
    float hStep_2 = _group_0_binding_0_vs.rEcefNed0_.w;
    float _e13 = _group_0_binding_0_vs.renderParams.w;
    float texSize = max(_e13, 2.0);
    int ix_1 = int(clamp(floor(((uvRing.x * (texSize - 1.0)) + 0.5)), 0.0, (texSize - 1.0)));
    int iy_1 = int(clamp(floor(((uvRing.y * (texSize - 1.0)) + 0.5)), 0.0, (texSize - 1.0)));
    if ((uvRing.z < 0.5)) {
        vec4 _e47 = texelFetch(_group_0_binding_1_vs, ivec2(ix_1, iy_1), 0);
        sampleH = _e47;
    } else {
        vec4 _e51 = texelFetch(_group_0_binding_2_vs, ivec2(ix_1, iy_1), 0);
        sampleH = _e51;
    }
    vec4 _e52 = sampleH;
    float _e53 = decodeHeight16_(_e52, hMin_2, hStep_2);
    float r2_ = ((neCoord.x * neCoord.x) + (neCoord.y * neCoord.y));
    float curvDropM = (r2_ / 12756274.0);
    float downM = (-(_e53) + curvDropM);
    vec4 _e67 = _group_0_binding_0_vs.originRteEcef;
    vec4 _e71 = _group_0_binding_0_vs.rEcefNed0_;
    vec4 _e78 = _group_0_binding_0_vs.rEcefNed1_;
    vec4 _e85 = _group_0_binding_0_vs.rEcefNed2_;
    vec3 posRte = (((_e67.xyz + (_e71.xyz * neCoord.x)) + (_e78.xyz * neCoord.y)) + (_e85.xyz * downM));
    mat4x4 _e93 = _group_0_binding_0_vs.viewProj;
    out_.pos = (_e93 * vec4(posRte, 1.0));
    out_.hWgs84_ = _e53;
    float _e102 = _group_0_binding_0_vs.rEcefNed1_.w;
    out_.clearanceM = (_e102 - _e53);
    out_.distM = length(posRte);
    out_.localNe = neCoord;
    out_.ringId = uvRing.z;
    out_.uv = uvRing.xy;
    VsOut _e111 = out_;
    gl_Position = _e111.pos;
    _vs2fs_location0 = _e111.hWgs84_;
    _vs2fs_location1 = _e111.clearanceM;
    _vs2fs_location2 = _e111.distM;
    _vs2fs_location3 = _e111.localNe;
    _vs2fs_location4 = _e111.ringId;
    _vs2fs_location5 = _e111.uv;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

