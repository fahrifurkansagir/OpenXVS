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
layout(std140) uniform Pass3Uniforms_block_0Fragment { Pass3Uniforms _group_0_binding_0_fs; };

smooth in vec4 _vs2fs_location0;
smooth in vec2 _vs2fs_location1;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    VsOut inp = VsOut(gl_FragCoord, _vs2fs_location0, _vs2fs_location1);
    vec3 rgb = vec3(0.0);
    float alpha = 0.0;
    rgb = inp.color.xyz;
    alpha = inp.color.w;
    float _e10 = _group_0_binding_0_fs.modeParams.x;
    if ((_e10 > 0.5)) {
        vec2 uv = inp.dashMeta;
        float r2_ = dot(uv, uv);
        float _e18 = _group_0_binding_0_fs.modeParams.y;
        float sigma = max(_e18, 0.5);
        float g = exp((-(sigma) * r2_));
        float _e27 = _group_0_binding_0_fs.modeParams.z;
        if ((g < _e27)) {
            discard;
        }
        float _e32 = _group_0_binding_0_fs.modeParams.x;
        if ((_e32 > 1.5)) {
            float core = exp((-11.0 * r2_));
            vec3 _e38 = rgb;
            rgb = clamp((_e38 + (vec3(0.55, 0.55, 0.55) * core)), vec3(0.0), vec3(1.0));
            float _e50 = alpha;
            alpha = clamp((_e50 * g), 0.0, 1.0);
        } else {
            float _e55 = alpha;
            alpha = clamp((_e55 * g), 0.0, 1.0);
        }
    } else {
        if ((inp.dashMeta.y > 0.001)) {
            float _e69 = _group_0_binding_0_fs.params.x;
            float phase = fract(((inp.dashMeta.x + (_e69 * 15.0)) / inp.dashMeta.y));
            if ((phase > 0.55)) {
                discard;
            }
        }
        if ((inp.dashMeta.y < -0.5)) {
            float _e86 = _group_0_binding_0_fs.params.x;
            float strobe = (0.45 + (0.55 * pow(max(sin(((_e86 * 9.0) + inp.dashMeta.x)), 0.0), 4.0)));
            vec3 _e101 = rgb;
            rgb = clamp((_e101 * (0.85 + (0.65 * strobe))), vec3(0.0), vec3(1.0));
            float _e112 = alpha;
            alpha = clamp((_e112 * (0.75 + (0.25 * strobe))), 0.0, 1.0);
        }
    }
    float _e124 = _group_0_binding_0_fs.params.y;
    if ((_e124 > 0.5)) {
        vec3 _e127 = rgb;
        float luma = dot(_e127, vec3(0.299, 0.587, 0.114));
        vec3 thermalBoost = mix(vec3(0.28, 0.95, 0.58), vec3(0.96, 1.0, 0.95), clamp((luma * 1.25), 0.0, 1.0));
        vec3 _e147 = rgb;
        rgb = mix(_e147, thermalBoost, 0.42);
    }
    vec3 _e150 = rgb;
    float _e151 = alpha;
    float _e155 = _group_0_binding_0_fs.params.w;
    _fs2p_location0 = vec4(_e150, clamp((_e151 * _e155), 0.0, 1.0));
    return;
}

