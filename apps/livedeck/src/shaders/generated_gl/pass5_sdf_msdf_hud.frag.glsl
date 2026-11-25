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
layout(std140) uniform Pass5Uniforms_block_0Fragment { Pass5Uniforms _group_0_binding_0_fs; };

uniform highp sampler2D _group_0_binding_1_fs;

smooth in vec2 _vs2fs_location0;
smooth in vec4 _vs2fs_location1;
smooth in vec4 _vs2fs_location2;
layout(location = 0) out vec4 _fs2p_location0;

float median3_(float a, float b, float c) {
    return max(min(a, b), min(max(a, b), c));
}

float sdRoundedBox(vec2 p, vec2 b_1, float r) {
    vec2 q = ((abs(p) - b_1) + vec2(r, r));
    return ((length(max(q, vec2(0.0, 0.0))) + min(max(q.x, q.y), 0.0)) - r);
}

void main() {
    VsOut inp = VsOut(gl_FragCoord, _vs2fs_location0, _vs2fs_location1, _vs2fs_location2);
    float alpha = 0.0;
    float kind = inp.primMeta.x;
    alpha = inp.color.w;
    if ((kind < 0.5)) {
        vec2 halfExt = vec2(inp.primMeta.y, inp.primMeta.z);
        vec2 pLocal = (inp.uv * halfExt);
        float _e25 = sdRoundedBox(pLocal, max((halfExt - vec2(0.75, 0.75)), vec2(0.25, 0.25)), inp.primMeta.w);
        float cov = (1.0 - smoothstep(-0.6, 0.6, _e25));
        float _e31 = alpha;
        alpha = (_e31 * cov);
    } else {
        if ((kind < 1.5)) {
            float radius = inp.primMeta.y;
            float strokeHalf = max(inp.primMeta.z, 0.6);
            vec2 pLocal_1 = (inp.uv * ((radius + strokeHalf) + 1.5));
            float dist = (abs((length(pLocal_1) - radius)) - strokeHalf);
            float cov_1 = (1.0 - smoothstep(-0.65, 0.65, dist));
            float _e55 = alpha;
            alpha = (_e55 * cov_1);
        } else {
            if ((kind < 2.5)) {
                vec4 _e63 = textureLod(_group_0_binding_1_fs, vec2(inp.uv), 0.0);
                vec3 sampleRgb = _e63.xyz;
                float _e68 = median3_(sampleRgb.x, sampleRgb.y, sampleRgb.z);
                float sd = (_e68 - 0.5);
                float _e74 = _group_0_binding_0_fs.hudParams.x;
                float pxRange = max(_e74, 1.0);
                float screenPxDist = ((sd * pxRange) * max(inp.primMeta.y, 0.75));
                float cov_2 = clamp((screenPxDist + 0.5), 0.0, 1.0);
                float _e88 = alpha;
                alpha = (_e88 * cov_2);
            } else {
                vec2 halfExt_1 = vec2(inp.primMeta.y, inp.primMeta.z);
                vec2 pLocal_2 = (inp.uv * halfExt_1);
                float _e107 = sdRoundedBox(pLocal_2, max((halfExt_1 - vec2(1.0, 1.0)), vec2(1.0, 1.0)), inp.primMeta.w);
                float cov_3 = (1.0 - smoothstep(-0.7, 0.7, _e107));
                float _e113 = alpha;
                alpha = (_e113 * cov_3);
            }
        }
    }
    float _e115 = alpha;
    if ((_e115 < 0.01)) {
        discard;
    }
    float _e120 = alpha;
    _fs2p_location0 = vec4(inp.color.xyz, _e120);
    return;
}

