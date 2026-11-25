#version 300 es

precision highp float;
precision highp int;

struct Pass5Uniforms {
    vec4 viewport;
    vec4 hudParams;
};
struct VsOut {
    vec4 pos;
    vec2 uv;
    vec4 color;
    vec4 primMeta;
};
layout(std140) uniform Pass5Uniforms_block_0Vertex { Pass5Uniforms _group_0_binding_0_vs; };

layout(location = 0) in vec2 _p2vs_location0;
layout(location = 1) in vec2 _p2vs_location1;
layout(location = 2) in vec4 _p2vs_location2;
layout(location = 3) in vec4 _p2vs_location3;
smooth out vec2 _vs2fs_location0;
smooth out vec4 _vs2fs_location1;
smooth out vec4 _vs2fs_location2;

float median3_(float a, float b, float c) {
    return max(min(a, b), min(max(a, b), c));
}

float sdRoundedBox(vec2 p, vec2 b_1, float r) {
    vec2 q = ((abs(p) - b_1) + vec2(r, r));
    return ((length(max(q, vec2(0.0, 0.0))) + min(max(q.x, q.y), 0.0)) - r);
}

void main() {
    vec2 posPx = _p2vs_location0;
    vec2 uv = _p2vs_location1;
    vec4 color = _p2vs_location2;
    vec4 primMeta = _p2vs_location3;
    VsOut out_ = VsOut(vec4(0.0), vec2(0.0), vec4(0.0), vec4(0.0));
    float _e7 = _group_0_binding_0_vs.viewport.x;
    float w = max(_e7, 1.0);
    float _e13 = _group_0_binding_0_vs.viewport.y;
    float h = max(_e13, 1.0);
    float xNdc = (((2.0 * posPx.x) / w) - 1.0);
    float yNdc = (1.0 - ((2.0 * posPx.y) / h));
    out_.pos = vec4(xNdc, yNdc, 0.01, 1.0);
    out_.uv = uv;
    out_.color = color;
    out_.primMeta = primMeta;
    VsOut _e36 = out_;
    gl_Position = _e36.pos;
    _vs2fs_location0 = _e36.uv;
    _vs2fs_location1 = _e36.color;
    _vs2fs_location2 = _e36.primMeta;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

