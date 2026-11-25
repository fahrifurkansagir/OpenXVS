#version 300 es

precision highp float;
precision highp int;

struct Pass3Uniforms {
    mat4x4 viewProj;
    vec4 params;
    vec4 modeParams;
};
struct VsOut {
    vec4 pos;
    vec4 color;
    vec2 dashMeta;
};
layout(std140) uniform Pass3Uniforms_block_0Vertex { Pass3Uniforms _group_0_binding_0_vs; };

layout(location = 0) in vec3 _p2vs_location0;
layout(location = 1) in vec4 _p2vs_location1;
layout(location = 2) in vec2 _p2vs_location2;
smooth out vec4 _vs2fs_location0;
smooth out vec2 _vs2fs_location1;

void main() {
    vec3 posRte = _p2vs_location0;
    vec4 color = _p2vs_location1;
    vec2 dashMeta = _p2vs_location2;
    VsOut out_ = VsOut(vec4(0.0), vec4(0.0), vec2(0.0));
    mat4x4 _e7 = _group_0_binding_0_vs.viewProj;
    out_.pos = (_e7 * vec4(posRte, 1.0));
    out_.color = color;
    out_.dashMeta = dashMeta;
    VsOut _e13 = out_;
    gl_Position = _e13.pos;
    _vs2fs_location0 = _e13.color;
    _vs2fs_location1 = _e13.dashMeta;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

