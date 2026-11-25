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
layout(std140) uniform Pass2Uniforms_block_0Fragment { Pass2Uniforms _group_0_binding_0_fs; };

uniform highp sampler2D _group_0_binding_1_fs;

uniform highp sampler2D _group_0_binding_2_fs;

uniform highp sampler2D _group_0_binding_3_fs;

uniform highp sampler2D _group_0_binding_4_fs;

uniform highp sampler2D _group_0_binding_6_fs;

smooth in float _vs2fs_location0;
smooth in float _vs2fs_location1;
smooth in float _vs2fs_location2;
smooth in vec2 _vs2fs_location3;
smooth in float _vs2fs_location4;
smooth in vec2 _vs2fs_location5;
layout(location = 0) out vec4 _fs2p_location0;

float decodeHeight16_(vec4 texel, float hMin, float hStep) {
    float hi = floor(((texel.x * 255.0) + 0.5));
    float lo = floor(((texel.y * 255.0) + 0.5));
    float q = ((hi * 256.0) + lo);
    return (hMin + (q * hStep));
}

float sampleDemClamped(bool isInner, int ix, int iy, int maxIdx, float hMin_1, float hStep_1) {
    int cx = min(max(ix, 0), maxIdx);
    int cy = min(max(iy, 0), maxIdx);
    vec4 tInner = texelFetch(_group_0_binding_1_fs, ivec2(cx, cy), 0);
    vec4 tOuter = texelFetch(_group_0_binding_2_fs, ivec2(cx, cy), 0);
    vec4 texel_1 = (isInner ? tInner : tOuter);
    float _e19 = decodeHeight16_(texel_1, hMin_1, hStep_1);
    return _e19;
}

void main() {
    VsOut inp = VsOut(gl_FragCoord, _vs2fs_location0, _vs2fs_location1, _vs2fs_location2, _vs2fs_location3, _vs2fs_location4, _vs2fs_location5);
    bool local = false;
    vec3 hypsoCol = vec3(0.0);
    vec3 waterCol = vec3(0.0);
    vec3 col = vec3(0.0);
    bool local_1 = false;
    vec3 lwirCol = vec3(0.0);
    bool local_2 = false;
    float hMslApprox = (inp.hWgs84_ + 32.63);
    vec2 uvClamped = clamp(inp.uv, vec2(0.0), vec2(1.0));
    bool isInner_1 = (inp.ringId < 0.5);
    vec4 _e16 = textureLod(_group_0_binding_3_fs, vec2(uvClamped), 0.0);
    vec3 orthoInnerRgb = _e16.xyz;
    vec4 _e21 = textureLod(_group_0_binding_4_fs, vec2(uvClamped), 0.0);
    vec3 orthoOuterRgb = _e21.xyz;
    float _e26 = _group_0_binding_0_fs.coreBoundsNe.z;
    float coreHalf = max(_e26, 1.0);
    float coreSpan = (coreHalf * 2.0);
    float _e36 = _group_0_binding_0_fs.coreBoundsNe.y;
    float coreU = ((inp.localNe.y - (_e36 - coreHalf)) / coreSpan);
    float _e43 = _group_0_binding_0_fs.coreBoundsNe.x;
    float coreV = (((_e43 + coreHalf) - inp.localNe.x) / coreSpan);
    vec2 coreUv = clamp(vec2(coreU, coreV), vec2(0.0), vec2(1.0));
    vec4 _e58 = textureLod(_group_0_binding_6_fs, vec2(coreUv), 0.0);
    vec3 orthoCoreRgb = _e58.xyz;
    float coreEdge = min(min(coreU, (1.0 - coreU)), min(coreV, (1.0 - coreV)));
    float _e73 = _group_0_binding_0_fs.sunDirNed.w;
    float coreMask = (smoothstep(0.0, 0.08, coreEdge) * clamp(_e73, 0.0, 1.0));
    vec3 baseOrthoRgb = (isInner_1 ? orthoInnerRgb : orthoOuterRgb);
    vec3 orthoRgb = mix(baseOrthoRgb, orthoCoreRgb, coreMask);
    float _e83 = _group_0_binding_0_fs.orthoParams.x;
    if ((_e83 > 0.5)) {
        float _e91 = _group_0_binding_0_fs.renderParams.y;
        local = (_e91 < 0.5);
    } else {
        local = false;
    }
    bool useOrtho = local;
    float _e99 = _group_0_binding_0_fs.orthoParams.y;
    float orthoBlend = (useOrtho ? clamp(_e99, 0.0, 1.0) : 0.0);
    float hMin_2 = _group_0_binding_0_fs.originRteEcef.w;
    float hStep_2 = _group_0_binding_0_fs.rEcefNed0_.w;
    float _e116 = _group_0_binding_0_fs.renderParams.w;
    float texSize = max(_e116, 2.0);
    int maxIdx_1 = int((texSize - 1.0));
    float fx = (uvClamped.x * (texSize - 1.0));
    float fy = (uvClamped.y * (texSize - 1.0));
    int ix_1 = int(floor(fx));
    int iy_1 = int(floor(fy));
    float tx = fract(fx);
    float ty = fract(fy);
    float _e139 = _group_0_binding_0_fs.orthoParams.w;
    float _e145 = _group_0_binding_0_fs.orthoParams.z;
    float ringExtentM = (isInner_1 ? max(_e145, 1000.0) : max(_e139, 1000.0));
    float cellM = (ringExtentM / (texSize - 1.0));
    float _e152 = sampleDemClamped(isInner_1, ix_1, iy_1, maxIdx_1, hMin_2, hStep_2);
    float _e155 = sampleDemClamped(isInner_1, (ix_1 + 1), iy_1, maxIdx_1, hMin_2, hStep_2);
    float _e158 = sampleDemClamped(isInner_1, ix_1, (iy_1 + 1), maxIdx_1, hMin_2, hStep_2);
    float _e163 = sampleDemClamped(isInner_1, (ix_1 + 1), (iy_1 + 1), maxIdx_1, hMin_2, hStep_2);
    float _e166 = sampleDemClamped(isInner_1, (ix_1 - 1), iy_1, maxIdx_1, hMin_2, hStep_2);
    float _e169 = sampleDemClamped(isInner_1, (ix_1 + 2), iy_1, maxIdx_1, hMin_2, hStep_2);
    float _e174 = sampleDemClamped(isInner_1, (ix_1 - 1), (iy_1 + 1), maxIdx_1, hMin_2, hStep_2);
    float _e179 = sampleDemClamped(isInner_1, (ix_1 + 2), (iy_1 + 1), maxIdx_1, hMin_2, hStep_2);
    float _e182 = sampleDemClamped(isInner_1, ix_1, (iy_1 - 1), maxIdx_1, hMin_2, hStep_2);
    float _e187 = sampleDemClamped(isInner_1, (ix_1 + 1), (iy_1 - 1), maxIdx_1, hMin_2, hStep_2);
    float _e190 = sampleDemClamped(isInner_1, ix_1, (iy_1 + 2), maxIdx_1, hMin_2, hStep_2);
    float _e195 = sampleDemClamped(isInner_1, (ix_1 + 1), (iy_1 + 2), maxIdx_1, hMin_2, hStep_2);
    float dE0_ = mix(((_e155 - _e166) * 0.5), ((_e169 - _e152) * 0.5), tx);
    float dE1_ = mix(((_e163 - _e174) * 0.5), ((_e179 - _e158) * 0.5), tx);
    float dH_dEast = (mix(dE0_, dE1_, ty) / cellM);
    float dN0_ = mix(((_e182 - _e158) * 0.5), ((_e152 - _e190) * 0.5), ty);
    float dN1_ = mix(((_e187 - _e163) * 0.5), ((_e155 - _e195) * 0.5), ty);
    float dH_dNorth = (mix(dN0_, dN1_, tx) / cellM);
    vec3 terrainNormal = normalize(vec3(-(dH_dNorth), -(dH_dEast), -1.0));
    float slopeMag = clamp((sqrt(((dH_dNorth * dH_dNorth) + (dH_dEast * dH_dEast))) * 1.4), 0.0, 0.55);
    vec4 _e244 = _group_0_binding_0_fs.sunDirNed;
    vec3 sunDir = normalize(_e244.xyz);
    float nDotL = max(dot(terrainNormal, sunDir), 0.0);
    float upFactor = clamp(((-(terrainNormal.z) * 0.5) + 0.5), 0.0, 1.0);
    vec3 hemiAmbient = mix(vec3(0.28, 0.3, 0.32), vec3(0.42, 0.49, 0.59), upFactor);
    vec3 sunLight = ((vec3(1.05, 0.99, 0.9) * nDotL) * 0.86);
    vec3 terrainIrradiance = ((hemiAmbient * 0.95) + sunLight);
    float microFade = (1.0 - smoothstep(140.0, 1800.0, inp.distM));
    float m1_ = (sin(((inp.localNe.x * 1.37) + (inp.localNe.y * 0.91))) * cos(((inp.localNe.x * 0.73) - (inp.localNe.y * 1.53))));
    float m2_ = (sin(((inp.localNe.x * 4.8) - (inp.localNe.y * 3.9))) * cos(((inp.localNe.x * 3.1) + (inp.localNe.y * 5.2))));
    float microDetail = (1.0 + ((((m1_ * 0.6) + (m2_ * 0.4)) * 0.055) * microFade));
    if ((hMslApprox <= 0.8)) {
        hypsoCol = vec3(0.045, 0.13, 0.21);
    } else {
        if ((hMslApprox < 45.0)) {
            float t = clamp(((hMslApprox - 0.8) / 44.2), 0.0, 1.0);
            hypsoCol = mix(vec3(0.15, 0.24, 0.19), vec3(0.24, 0.33, 0.22), t);
        } else {
            if ((hMslApprox < 250.0)) {
                float t_1 = clamp(((hMslApprox - 45.0) / 205.0), 0.0, 1.0);
                hypsoCol = mix(vec3(0.24, 0.33, 0.22), vec3(0.39, 0.35, 0.26), t_1);
            } else {
                float t_2 = clamp(((hMslApprox - 250.0) / 350.0), 0.0, 1.0);
                hypsoCol = mix(vec3(0.39, 0.35, 0.26), vec3(0.56, 0.5, 0.44), t_2);
            }
        }
    }
    vec3 _e395 = hypsoCol;
    vec3 landAlbedo = (mix(_e395, orthoRgb, orthoBlend) * microDetail);
    vec3 litLand = clamp((landAlbedo * terrainIrradiance), vec3(0.0), vec3(1.0));
    float tSim = _group_0_binding_0_fs.renderParams.z;
    float _e411 = _group_0_binding_0_fs.camPosNeDown.x;
    float _e418 = _group_0_binding_0_fs.camPosNeDown.y;
    float _e425 = _group_0_binding_0_fs.camPosNeDown.z;
    vec3 toCamNed = vec3((_e411 - inp.localNe.x), (_e418 - inp.localNe.y), (_e425 + inp.hWgs84_));
    vec3 viewDir = normalize(toCamNed);
    float waveFade = (smoothstep(2600.0, 120.0, inp.distM) * clamp((abs(viewDir.z) * 2.8), 0.06, 1.0));
    float p1_ = (((inp.localNe.x * 0.035) + (inp.localNe.y * 0.048)) + (tSim * 0.85));
    float p2_ = (((inp.localNe.x * 0.062) - (inp.localNe.y * 0.054)) - (tSim * 1.15));
    float p3_ = (((inp.localNe.x * 0.095) + (inp.localNe.y * 0.082)) + (tSim * 1.45));
    float wN = ((((cos(p1_) * 0.018) + (cos(p2_) * 0.013)) + (cos(p3_) * 0.007)) * waveFade);
    float wE = ((((cos(p1_) * 0.022) - (cos(p2_) * 0.012)) + (cos(p3_) * 0.006)) * waveFade);
    vec3 waterNormal = normalize(vec3(wN, wE, -1.0));
    float nDotV = max(dot(waterNormal, viewDir), 0.001);
    float fresnel = (0.02 + (0.98 * pow((1.0 - nDotV), 5.0)));
    vec3 reflDir = reflect(-(viewDir), waterNormal);
    float reflElev = clamp(-(reflDir.z), 0.0, 1.0);
    vec3 skyRefl = mix(vec3(0.68, 0.77, 0.86), vec3(0.1, 0.25, 0.5), pow(reflElev, 0.55));
    vec3 deepBayCol = mix(vec3(0.025, 0.085, 0.155), (orthoRgb * 0.58), (orthoBlend * 0.16));
    waterCol = mix(deepBayCol, skyRefl, clamp(((fresnel * 0.88) + 0.06), 0.0, 0.94));
    vec3 halfVec = normalize((sunDir + viewDir));
    float nDotH = max(dot(waterNormal, halfVec), 0.0);
    float a2_ = (0.11 * 0.11);
    float ggxDenom = (((nDotH * nDotH) * (a2_ - 1.0)) + 1.0);
    float ndf = (a2_ / (((3.1415927 * ggxDenom) * ggxDenom) + 0.0001));
    float waterSunVis = max(dot(waterNormal, sunDir), 0.0);
    float specGlint = min((((ndf * fresnel) * waterSunVis) * (0.28 + (0.26 * waveFade))), 1.6);
    vec3 _e583 = waterCol;
    waterCol = (_e583 + (vec3(1.0, 0.92, 0.78) * specGlint));
    float landMask = smoothstep(0.15, 1.45, hMslApprox);
    vec3 _e593 = waterCol;
    col = mix(_e593, litLand, landMask);
    float contour = (1.0 - smoothstep(0.0, 1.8, (abs((fract(((hMslApprox / 50.0) + 0.5)) - 0.5)) * 50.0)));
    float gx = (abs((fract(((inp.localNe.x / 1000.0) + 0.5)) - 0.5)) * 1000.0);
    float gy = (abs((fract(((inp.localNe.y / 1000.0) + 0.5)) - 0.5)) * 1000.0);
    float gridLine = (1.0 - smoothstep(0.0, 16.0, min(gx, gy)));
    float overlayStrength = mix(1.0, 0.18, orthoBlend);
    vec3 _e644 = col;
    col = mix(_e644, vec3(0.2, 0.46, 0.42), (((gridLine * 0.1) + (contour * 0.12)) * overlayStrength));
    if ((hMslApprox > 12.0)) {
        local_1 = (inp.distM < 2200.0);
    } else {
        local_1 = false;
    }
    bool _e664 = local_1;
    if (_e664) {
        float _e668 = _group_0_binding_0_fs.renderParams.z;
        float pulse = (0.65 + (0.35 * sin((_e668 * 6.0))));
        if ((inp.clearanceM < 60.0)) {
            vec3 _e679 = col;
            col = mix(_e679, vec3(0.92, 0.14, 0.12), (0.72 * pulse));
        } else {
            if ((inp.clearanceM < 150.0)) {
                vec3 _e690 = col;
                col = mix(_e690, vec3(0.95, 0.66, 0.1), 0.48);
            }
        }
    }
    float _e700 = _group_0_binding_0_fs.rEcefNed2_.w;
    float fogContrast = clamp(_e700, 0.08, 1.0);
    float fogDensity = (1.0 - fogContrast);
    float distFactor = (1.0 - exp((-(inp.distM) * (2.8e-5 + (fogDensity * 0.00055)))));
    vec3 fogCol = mix(vec3(0.68, 0.77, 0.86), vec3(0.64, 0.69, 0.73), fogDensity);
    float _e728 = _group_0_binding_0_fs.renderParams.x;
    if ((_e728 > 0.5)) {
        float orthoLuma = dot(orthoRgb, vec3(0.299, 0.587, 0.114));
        bool isBay = (hMslApprox <= 0.8);
        float orthoThermalMod = (useOrtho ? ((orthoLuma - 0.32) * 0.42) : 0.0);
        float baseThermal = (isBay ? clamp(((0.11 + (0.04 * contour)) + (useOrtho ? (orthoLuma * 0.1) : 0.0)), 0.06, 0.26) : clamp((((0.42 + clamp((hMslApprox / 520.0), 0.0, 0.34)) + slopeMag) + orthoThermalMod), 0.28, 0.98));
        float lwirAtm = (1.0 - exp((-(inp.distM) * 3.2e-5)));
        float thermalVal = mix(baseThermal, 0.24, (lwirAtm * 0.38));
        vec3 coolCol = vec3(0.02, 0.13, 0.09);
        vec3 midCol = vec3(0.2, 0.9, 0.5);
        vec3 hotCol = vec3(0.93, 1.0, 0.94);
        lwirCol = ((thermalVal < 0.58) ? mix(coolCol, midCol, clamp((thermalVal / 0.58), 0.0, 1.0)) : mix(midCol, hotCol, clamp(((thermalVal - 0.58) / 0.42), 0.0, 1.0)));
        vec3 _e810 = lwirCol;
        lwirCol = (_e810 + (vec3(0.18, 0.38, 0.24) * ((contour * 0.3) + (gridLine * 0.14))));
        if ((hMslApprox > 12.0)) {
            local_2 = (inp.distM < 2200.0);
        } else {
            local_2 = false;
        }
        bool _e830 = local_2;
        if (_e830) {
            float _e834 = _group_0_binding_0_fs.renderParams.z;
            float pulse_1 = (0.65 + (0.35 * sin((_e834 * 6.0))));
            if ((inp.clearanceM < 60.0)) {
                vec3 _e845 = lwirCol;
                lwirCol = mix(_e845, vec3(1.0, 0.2, 0.18), (0.72 * pulse_1));
            } else {
                if ((inp.clearanceM < 150.0)) {
                    vec3 _e856 = lwirCol;
                    lwirCol = mix(_e856, vec3(1.0, 0.72, 0.14), 0.5);
                }
            }
        }
        vec3 _e863 = lwirCol;
        col = _e863;
    } else {
        vec3 _e864 = col;
        col = mix(_e864, fogCol, clamp(distFactor, 0.0, 0.94));
    }
    vec3 _e869 = col;
    _fs2p_location0 = vec4(clamp(_e869, vec3(0.0), vec3(1.0)), 1.0);
    return;
}

