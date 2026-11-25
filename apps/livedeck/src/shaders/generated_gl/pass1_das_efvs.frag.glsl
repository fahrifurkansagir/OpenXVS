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
layout(std140) uniform Pass1Uniforms_block_0Fragment { Pass1Uniforms _group_0_binding_0_fs; };

smooth in vec2 _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

bool bitEnabled(float maskF, int bitIdx) {
    uint m = uint(max((maskF + 0.5), 0.0));
    return (((m >> uint(bitIdx)) & 1u) != 0u);
}

void main() {
    VsOut inp = VsOut(gl_FragCoord, _vs2fs_location0);
    vec3 rayBody = vec3(0.0);
    int camIdx = 0;
    float seamMetric = 1.0;
    bool local = false;
    bool local_1 = false;
    bool local_2 = false;
    vec3 color = vec3(0.0);
    float fx = _group_0_binding_0_fs.rBodyCam0_.w;
    float fy = _group_0_binding_0_fs.rBodyCam1_.w;
    float cx = _group_0_binding_0_fs.rBodyCam2_.w;
    float cy = _group_0_binding_0_fs.rNedBody0_.w;
    float px = (inp.pos.x - 0.5);
    float py = (inp.pos.y - 0.5);
    vec3 rayCam = normalize(vec3(((px - cx) / fx), ((py - cy) / fy), 1.0));
    vec4 _e34 = _group_0_binding_0_fs.rBodyCam0_;
    vec4 _e38 = _group_0_binding_0_fs.rBodyCam1_;
    vec4 _e42 = _group_0_binding_0_fs.rBodyCam2_;
    mat3x3 rBodyCam = mat3x3(_e34.xyz, _e38.xyz, _e42.xyz);
    vec4 _e47 = _group_0_binding_0_fs.rNedBody0_;
    vec4 _e51 = _group_0_binding_0_fs.rNedBody1_;
    vec4 _e55 = _group_0_binding_0_fs.rNedBody2_;
    mat3x3 rNedBody = mat3x3(_e47.xyz, _e51.xyz, _e55.xyz);
    rayBody = normalize((rBodyCam * rayCam));
    float _e62 = rayBody.x;
    float ax = abs(_e62);
    float _e65 = rayBody.y;
    float ay = abs(_e65);
    float _e68 = rayBody.z;
    float az = abs(_e68);
    if ((ax >= ay)) {
        local = (ax >= az);
    } else {
        local = false;
    }
    bool _e79 = local;
    if (_e79) {
        float _e81 = rayBody.x;
        camIdx = ((_e81 >= 0.0) ? 0 : 2);
        seamMetric = (ax - max(ay, az));
    } else {
        if ((ay >= ax)) {
            local_1 = (ay >= az);
        } else {
            local_1 = false;
        }
        bool _e94 = local_1;
        if (_e94) {
            float _e96 = rayBody.y;
            camIdx = ((_e96 >= 0.0) ? 1 : 3);
            seamMetric = (ay - max(ax, az));
        } else {
            float _e105 = rayBody.z;
            camIdx = ((_e105 >= 0.0) ? 5 : 4);
            seamMetric = (az - max(ax, ay));
        }
    }
    float _e116 = _group_0_binding_0_fs.healthParams.x;
    int _e117 = camIdx;
    bool _e118 = bitEnabled(_e116, _e117);
    float camRBias = _group_0_binding_0_fs.envParams.z;
    int _e123 = camIdx;
    if ((_e123 == 1)) {
        local_2 = (abs(camRBias) > 1e-5);
    } else {
        local_2 = false;
    }
    bool _e132 = local_2;
    if (_e132) {
        float cb = cos(camRBias);
        float sb = sin(camRBias);
        float _e136 = rayBody.x;
        float _e138 = rayBody.y;
        float _e141 = rayBody.z;
        float _e145 = rayBody.y;
        float _e148 = rayBody.z;
        rayBody = normalize(vec3(_e136, ((_e138 * cb) - (_e141 * sb)), ((_e145 * sb) + (_e148 * cb))));
    }
    vec3 _e153 = rayBody;
    vec3 rayNed = normalize((rNedBody * _e153));
    float elev = -(asin(clamp(rayNed.z, -1.0, 1.0)));
    float horizonDip = _group_0_binding_0_fs.sunDirNed.w;
    float elevAboveLimb = (elev + horizonDip);
    float _e170 = _group_0_binding_0_fs.envParams.x;
    float fogContrast = clamp(_e170, 0.08, 1.0);
    float efvs = _group_0_binding_0_fs.envParams.y;
    vec4 _e180 = _group_0_binding_0_fs.sunDirNed;
    vec3 sunDir = normalize(_e180.xyz);
    float cosTheta = clamp(dot(rayNed, sunDir), -1.0, 1.0);
    float glare = _group_0_binding_0_fs.healthParams.y;
    float sunDisc = ((smoothstep(0.99935, 0.99988, cosTheta) * 1.65) * glare);
    float _e200 = _group_0_binding_0_fs.healthParams.z;
    float altKm = clamp((_e200 * 0.001), 0.0, 20.0);
    float atmDensity = exp((-(altKm) / 8.4));
    vec3 rayleighHorizon = vec3(0.68, 0.77, 0.86);
    if ((elevAboveLimb >= 0.0)) {
        float elDeg = (elevAboveLimb * 57.29578);
        float airMass = min((1.0 / (sin(elevAboveLimb) + (0.15 * pow((elDeg + 3.885), -1.253)))), 32.0);
        float sunElDeg = max((-(asin(clamp(sunDir.z, -1.0, 1.0))) * 57.29578), 1.5);
        float sunAirMass = min((1.0 / (sin((sunElDeg * 0.01745329)) + (0.15 * pow((sunElDeg + 3.885), -1.253)))), 28.0);
        vec3 betaR = vec3(0.058, 0.135, 0.331);
        vec3 tauR = (((betaR * airMass) * atmDensity) * 0.38);
        vec3 sunTrans = exp(((-(betaR) * sunAirMass) * 0.22));
        float phaseR = (0.75 * (1.0 + (cosTheta * cosTheta)));
        vec3 rayleighScatter = ((((vec3(1.0) - exp(-(tauR))) * phaseR) * sunTrans) * vec3(0.9, 0.96, 1.12));
        vec3 zenithBlue = (vec3(0.07, 0.21, 0.46) * mix(0.62, 1.0, atmDensity));
        float horizonBand = exp((-(elevAboveLimb) * 7.5));
        color = (zenithBlue + (rayleighScatter * 0.62));
        vec3 _e300 = color;
        color = mix(_e300, rayleighHorizon, (horizonBand * 0.78));
        float g2_ = (0.76 * 0.76);
        float hgDenom = max(((1.0 + g2_) - ((2.0 * 0.76) * cosTheta)), 0.015);
        float phaseM = ((1.0 - g2_) / (12.566371 * pow(hgDenom, 1.5)));
        float mieOptical = (1.0 - exp(((-0.085 * airMass) * atmDensity)));
        float mieAureole = (min(((phaseM * 0.38) * mieOptical), 1.6) * glare);
        float warmHorizonGlow = (((pow(max(cosTheta, 0.0), 3.0) * horizonBand) * 0.18) * glare);
        vec3 _e341 = color;
        color = (_e341 + (vec3(1.0, 0.91, 0.76) * ((mieAureole + warmHorizonGlow) + sunDisc)));
        float azim = atan(rayNed.y, rayNed.x);
        float _e356 = _group_0_binding_0_fs.envParams.w;
        float drift = (_e356 * 0.004);
        float c1_ = (sin((((azim * 7.0) + (elevAboveLimb * 32.0)) + drift)) * cos((((azim * 13.0) - (elevAboveLimb * 18.0)) - (drift * 0.7))));
        float c2_ = (sin((((azim * 19.0) - (elevAboveLimb * 48.0)) + (drift * 1.3))) * 0.5);
        float cirrusVal = smoothstep(0.18, 0.86, (((c1_ + c2_) * 0.42) + 0.48));
        float cloudMask = (smoothstep(0.05, 0.32, elevAboveLimb) * smoothstep(0.78, 0.24, elevAboveLimb));
        vec3 cirrusCol = mix(vec3(0.86, 0.9, 0.95), vec3(1.0, 0.94, 0.86), (pow(max(cosTheta, 0.0), 4.0) * 0.6));
        vec3 _e417 = color;
        color = mix(_e417, cirrusCol, ((cirrusVal * cloudMask) * 0.16));
        vec3 _e422 = color;
        vec3 _e425 = color;
        color = (_e422 / (vec3(1.0) + (max((_e425 - vec3(0.88)), vec3(0.0)) * 0.55)));
    } else {
        float dep = clamp((-(elevAboveLimb) * 4.5), 0.0, 1.0);
        vec3 bayCol = vec3(0.045, 0.125, 0.205);
        color = mix((rayleighHorizon * 0.88), bayCol, pow(dep, 0.58));
        float glintMask = (((pow(max(cosTheta, 0.0), 18.0) * exp((-(dep) * 2.2))) * 0.26) * glare);
        vec3 _e463 = color;
        color = (_e463 + (vec3(1.0, 0.92, 0.76) * glintMask));
    }
    float fogDensity = (1.0 - fogContrast);
    float fogBand = exp((-(abs(elevAboveLimb)) * 10.0));
    vec3 fogColor = vec3(0.64, 0.69, 0.73);
    vec3 _e481 = color;
    color = mix(_e481, fogColor, clamp((fogDensity * (0.68 + (0.32 * fogBand))), 0.0, 0.92));
    if ((efvs > 0.5)) {
        float skyCold = mix(0.16, 0.02, clamp(pow((max(elevAboveLimb, 0.0) * 2.4), 0.65), 0.0, 1.0));
        float groundWarm = mix(0.28, 0.54, clamp((-(elevAboveLimb) * 3.5), 0.0, 1.0));
        float limbGlow = (exp((-(abs(elevAboveLimb)) * 42.0)) * 0.22);
        float limbContrast = ((elevAboveLimb >= 0.0) ? skyCold : groundWarm);
        float _e529 = _group_0_binding_0_fs.envParams.w;
        float scan = (0.96 + (0.04 * sin(((py * 1.5) + (_e529 * 8.0)))));
        float thermalLuma = clamp((((limbContrast + limbGlow) + (sunDisc * 0.75)) * scan), 0.02, 0.98);
        vec3 coolCol = vec3(0.02, 0.14, 0.09);
        vec3 midCol = vec3(0.18, 0.86, 0.48);
        vec3 hotCol = vec3(0.92, 1.0, 0.94);
        if ((thermalLuma < 0.55)) {
            color = mix(coolCol, midCol, (thermalLuma / 0.55));
        } else {
            color = mix(midCol, hotCol, ((thermalLuma - 0.55) / 0.45));
        }
    }
    float _e568 = seamMetric;
    float seamLine = (1.0 - smoothstep(0.0, 0.008, _e568));
    vec3 _e574 = color;
    color = mix(_e574, vec3(0.22, 0.48, 0.52), (seamLine * 0.22));
    if (!(_e118)) {
        float hatch = step(0.5, fract(((px + py) * 0.06)));
        vec3 maskCol = mix(vec3(0.14, 0.04, 0.04), vec3(0.42, 0.09, 0.06), hatch);
        vec3 _e598 = color;
        color = mix((_e598 * 0.18), maskCol, 0.86);
    }
    vec3 _e603 = color;
    _fs2p_location0 = vec4(clamp(_e603, vec3(0.0), vec3(1.0)), 1.0);
    return;
}

