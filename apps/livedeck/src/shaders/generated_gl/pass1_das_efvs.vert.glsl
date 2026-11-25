#version 300 es

precision highp float;
precision highp int;

struct Pass1Uniforms {
    vec4 rBodyCam0_;
    vec4 rBodyCam1_;
    vec4 rBodyCam2_;
    vec4 rNedBody0_;
    vec4 rNedBody1_;
    vec4 rNedBody2_;
    vec4 sunDirNed;
    vec4 envParams;
    vec4 healthParams;
};
struct VsOut {
    vec4 pos;
    vec2 ndc;
};
layout(location = 0) in vec2 _p2vs_location0;
smooth out vec2 _vs2fs_location0;

bool bitEnabled(float maskF, int bitIdx) {
    uint m = uint(max((maskF + 0.5), 0.0));
    return (((m >> uint(bitIdx)) & 1u) != 0u);
}

void main() {
    vec2 posNdc = _p2vs_location0;
    VsOut out_ = VsOut(vec4(0.0), vec2(0.0));
    out_.pos = vec4(posNdc, 0.9999, 1.0);
    out_.ndc = posNdc;
    VsOut _e7 = out_;
    gl_Position = _e7.pos;
    _vs2fs_location0 = _e7.ndc;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

