/**
 * Precision 3D Aerospace Safety Symbology (`OpenXVS Studio`).
 *
 * Renders authoritative Anduril Lattice / Garmin Synthetic Vision 3D avionics symbology:
 *   1. `XVS-ActiveRange` Sleek Translucent Sensor FOV Cone (`opacity ~0.07–0.10`, normal alpha
 *      blending with crisp edge hairlines) + Ultra-Clean Segmented Thin Range-Uncertainty Rings.
 *   2. Crisp 3D Tactical Target Diamond & 8-Corner Bracket Wireframe around `CBDR-1` + Thin
 *      Segmented `SC3BF` Separation Perimeter Ring (zero opaque pastel game-shield bubble).
 *   3. Crisp 3D Planned `<=12°` S-Swerve Avoidance Trajectory Corridor Arc with animated chevrons.
 *   4. `XVS-ForcedLand` 3D Wind-Warped Trochoid Highway Ribbon + Holographic Gates.
 *   5. Optional `6-Cam` Translucent Fresnel Optical Frustums + Azimuth Dome around ownship.
 *   6. `GPS Spoof` Quarantine Shield Ring, Ghost Trajectory Marker & `XVS-Nav` Ground-Lock Rays.
 */

import * as THREE from "three";
import {
  ActiveRangeModeTs,
  R_SEP_M,
  type StudioSimSnapshot,
} from "../sim/studio_sim.ts";

export interface SafetyManifoldsRig {
  readonly group: THREE.Group;
  setSensorDomeVisible(visible: boolean): void;
  isSensorDomeVisible(): boolean;
  update(snap: StudioSimSnapshot, simTimeS: number): void;
}

const COLOR_ALERT_CRIMSON = 0xef4444;
const COLOR_CAUTION_AMBER = 0xf59e0b;
const COLOR_NOMINAL_CYAN = 0x38bdf8;
const COLOR_DEEP_CYAN = 0x0284c7;
const COLOR_EMERALD_LOCK = 0x10b981;
const COLOR_EMERALD_BRIGHT = 0x34d399;

function createFresnelVolumeMaterial(colorHex: number, opacity: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uColor: { value: new THREE.Color(colorHex) },
      uOpacity: { value: opacity },
      uTime: { value: 0.0 },
    },
    vertexShader: `
      varying vec3 vNormalW;
      varying vec3 vViewDir;
      varying vec2 vUv;
      void main() {
        vUv = uv;
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vNormalW = normalize(mat3(modelMatrix) * normal);
        vViewDir = normalize(cameraPosition - worldPos.xyz);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform float uTime;
      varying vec3 vNormalW;
      varying vec3 vViewDir;
      varying vec2 vUv;
      void main() {
        float fresnel = pow(1.0 - abs(dot(normalize(vNormalW), normalize(vViewDir))), 2.1);
        float scan = 0.88 + 0.12 * sin(vUv.y * 24.0 - uTime * 3.2);
        float alpha = clamp((0.12 + 0.88 * fresnel) * uOpacity * scan, 0.0, 0.85);
        gl_FragColor = vec4(uColor * (1.1 + 0.45 * fresnel), alpha);
      }
    `,
  });
}

/** Ultra-clean normal-blended translucent FOV cone so runway/Bay terrain stays razor sharp through it. */
function createSubtleFovConeMaterial(colorHex: number, opacity: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.NormalBlending,
    uniforms: {
      uColor: { value: new THREE.Color(colorHex) },
      uOpacity: { value: opacity },
      uTime: { value: 0.0 },
    },
    vertexShader: `
      varying vec3 vNormalW;
      varying vec3 vViewDir;
      varying vec2 vUv;
      void main() {
        vUv = uv;
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vNormalW = normalize(mat3(modelMatrix) * normal);
        vViewDir = normalize(cameraPosition - worldPos.xyz);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform float uTime;
      varying vec3 vNormalW;
      varying vec3 vViewDir;
      varying vec2 vUv;
      void main() {
        float edge = pow(1.0 - abs(dot(normalize(vNormalW), normalize(vViewDir))), 1.75);
        float endFade = smoothstep(0.0, 0.14, vUv.y) * smoothstep(1.0, 0.82, vUv.y);
        float sweep = 0.90 + 0.10 * sin(vUv.y * 20.0 - uTime * 2.6);
        float alpha = clamp((0.25 + 0.75 * edge) * uOpacity * endFade * sweep, 0.0, 0.18);
        gl_FragColor = vec4(uColor, alpha);
      }
    `,
  });
}

/** Builds a crisp 1px segmented hairline ring in the XY plane with optional cardinal ticks. */
function createSegmentedRingGeometry(
  radius: number,
  segments: number,
  duty: number,
  tickFrac = 0.0,
): THREE.BufferGeometry {
  const pts: number[] = [];
  const step = (Math.PI * 2) / segments;
  const subSteps = 3;
  for (let i = 0; i < segments; i++) {
    const a0 = i * step;
    const aSpan = step * duty;
    for (let s = 0; s < subSteps; s++) {
      const t0 = a0 + (aSpan * s) / subSteps;
      const t1 = a0 + (aSpan * (s + 1)) / subSteps;
      pts.push(
        Math.cos(t0) * radius,
        Math.sin(t0) * radius,
        0,
        Math.cos(t1) * radius,
        Math.sin(t1) * radius,
        0,
      );
    }
  }
  if (tickFrac > 0) {
    const rIn = radius * (1.0 - tickFrac * 0.55);
    const rOut = radius * (1.0 + tickFrac);
    for (let k = 0; k < 4; k++) {
      const a = (k * Math.PI) / 2;
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      pts.push(ca * rIn, sa * rIn, 0, ca * rOut, sa * rOut, 0);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return geo;
}

/** Builds a precision 3D tactical target diamond wireframe (octahedron + outer axis ticks). */
function createTacticalDiamondGeometry(radius = 1.0): THREE.BufferGeometry {
  const r = radius;
  const px: [number, number, number] = [r, 0, 0];
  const nx: [number, number, number] = [-r, 0, 0];
  const py: [number, number, number] = [0, r * 1.08, 0];
  const ny: [number, number, number] = [0, -r * 1.08, 0];
  const pz: [number, number, number] = [0, 0, r];
  const nz: [number, number, number] = [0, 0, -r];

  const edges: Array<[[number, number, number], [number, number, number]]> = [
    // Top pyramid
    [py, px],
    [py, nx],
    [py, pz],
    [py, nz],
    // Bottom pyramid
    [ny, px],
    [ny, nx],
    [ny, pz],
    [ny, nz],
    // Equatorial diamond
    [px, pz],
    [pz, nx],
    [nx, nz],
    [nz, px],
    // Crisp precision outer tick markers
    [
      [r * 1.08, 0, 0],
      [r * 1.32, 0, 0],
    ],
    [
      [-r * 1.08, 0, 0],
      [-r * 1.32, 0, 0],
    ],
    [
      [0, r * 1.15, 0],
      [0, r * 1.38, 0],
    ],
    [
      [0, -r * 1.15, 0],
      [0, -r * 1.38, 0],
    ],
  ];
  const pts: number[] = [];
  for (const [a, b] of edges) {
    pts.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return geo;
}

/** Builds 8 crisp 3D tactical corner brackets framing the intruder aircraft. */
function createCornerBrackets3DGeometry(halfSize = 1.0, armFrac = 0.36): THREE.BufferGeometry {
  const s = halfSize;
  const arm = halfSize * armFrac;
  const pts: number[] = [];
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const cx = sx * s;
        const cy = sy * s;
        const cz = sz * s;
        pts.push(cx, cy, cz, cx - sx * arm, cy, cz);
        pts.push(cx, cy, cz, cx, cy - sy * arm, cz);
        pts.push(cx, cy, cz, cx, cy, cz - sz * arm);
      }
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return geo;
}

/** Builds 4 crisp longitudinal FOV frustum edge hairlines + 4 corner aperture arcs along +Z [0..1]. */
function createFovConeEdgesGeometry(): THREE.BufferGeometry {
  const pts: number[] = [];
  const apexR = 0.035;
  const baseR = 1.0;
  const angles = [Math.PI * 0.25, Math.PI * 0.75, Math.PI * 1.25, Math.PI * 1.75];
  for (const ang of angles) {
    const ca = Math.cos(ang);
    const sa = Math.sin(ang);
    pts.push(ca * apexR, sa * apexR, 0.0, ca * baseR, sa * baseR, 1.0);
    // Crisp corner aperture arc around each frustum edge at base z = 1.0
    const span = 0.16;
    for (let k = -2; k < 2; k++) {
      const a0 = ang + (k / 2) * span;
      const a1 = ang + ((k + 1) / 2) * span;
      pts.push(
        Math.cos(a0) * baseR,
        Math.sin(a0) * baseR,
        1.0,
        Math.cos(a1) * baseR,
        Math.sin(a1) * baseR,
        1.0,
      );
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return geo;
}

export function createSafetyManifoldsRig(): SafetyManifoldsRig {
  const group = new THREE.Group();
  group.name = "openxvs_safety_manifolds";

  // ---------------------------------------------------------------------------
  // 1. XVS-ActiveRange Sleek Sensor FOV Cone + Thin Segmented Range-Uncertainty Rings
  // ---------------------------------------------------------------------------
  const arGroup = new THREE.Group();
  arGroup.name = "active_range_fov_rig";
  group.add(arGroup);

  // Unit frustum aligned along +Z from z=0 (ownship apex) to z=1 (base at rHat+sigma)
  const coneGeo = new THREE.CylinderGeometry(1.0, 0.035, 1.0, 32, 1, true);
  coneGeo.rotateX(Math.PI * 0.5);
  coneGeo.translate(0, 0, 0.5);
  const coneMat = createSubtleFovConeMaterial(COLOR_CAUTION_AMBER, 0.085);
  const coneMesh = new THREE.Mesh(coneGeo, coneMat);
  arGroup.add(coneMesh);

  // Crisp 4-ray frustum edge hairlines + corner aperture ticks
  const coneEdgesGeo = createFovConeEdgesGeometry();
  const coneEdgesMat = new THREE.LineBasicMaterial({
    color: COLOR_CAUTION_AMBER,
    transparent: true,
    opacity: 0.36,
    depthWrite: false,
  });
  const coneEdges = new THREE.LineSegments(coneEdgesGeo, coneEdgesMat);
  arGroup.add(coneEdges);

  // Thin precision LOS boresight hairline from nose radome to target bracket perimeter
  const losLineGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, 1),
  ]);
  const losLineMat = new THREE.LineBasicMaterial({
    color: COLOR_NOMINAL_CYAN,
    transparent: true,
    opacity: 0.44,
    depthWrite: false,
  });
  const losLine = new THREE.Line(losLineGeo, losLineMat);
  arGroup.add(losLine);

  // Primary ultra-clean segmented thin range-uncertainty ring at rHat (no scaled Torus discs!)
  const rangeHatRingGeo = createSegmentedRingGeometry(1.0, 28, 0.58, 0.11);
  const rangeHatRingMat = new THREE.LineBasicMaterial({
    color: COLOR_NOMINAL_CYAN,
    transparent: true,
    opacity: 0.44,
    depthWrite: false,
  });
  const rangeHatRing = new THREE.LineSegments(rangeHatRingGeo, rangeHatRingMat);
  arGroup.add(rangeHatRing);

  // Subtle 4-arc conformal lower-bound bracket at rSafeLower (low visual clutter)
  const rangeLowerRingGeo = createSegmentedRingGeometry(1.0, 4, 0.28, 0.07);
  const rangeLowerRingMat = new THREE.LineBasicMaterial({
    color: COLOR_CAUTION_AMBER,
    transparent: true,
    opacity: 0.38,
    depthWrite: false,
  });
  const rangeLowerRing = new THREE.LineSegments(rangeLowerRingGeo, rangeLowerRingMat);
  arGroup.add(rangeLowerRing);

  // ---------------------------------------------------------------------------
  // 2. Crisp 3D Tactical Target Diamond / Bracket Wireframe + Thin SC3BF Perimeter Ring
  // ---------------------------------------------------------------------------
  const sc3bfGroup = new THREE.Group();
  sc3bfGroup.name = "tactical_target_designator";
  group.add(sc3bfGroup);

  const targetDiamondMat = new THREE.LineBasicMaterial({
    color: COLOR_CAUTION_AMBER,
    transparent: true,
    opacity: 0.94,
    depthWrite: false,
  });
  const targetDiamond = new THREE.LineSegments(
    createTacticalDiamondGeometry(1.0),
    targetDiamondMat,
  );
  sc3bfGroup.add(targetDiamond);

  // Subtle nested contrast wireframe so target reticle stays razor-sharp against bright daylight sky
  const targetInnerDiamondMat = new THREE.LineBasicMaterial({
    color: COLOR_DEEP_CYAN,
    transparent: true,
    opacity: 0.78,
    depthWrite: false,
  });
  const targetInnerDiamond = new THREE.LineSegments(
    createTacticalDiamondGeometry(0.92),
    targetInnerDiamondMat,
  );
  sc3bfGroup.add(targetInnerDiamond);

  const targetBracketsMat = new THREE.LineBasicMaterial({
    color: COLOR_CAUTION_AMBER,
    transparent: true,
    opacity: 0.98,
    depthWrite: false,
  });
  const targetBrackets = new THREE.LineSegments(
    createCornerBrackets3DGeometry(1.0, 0.36),
    targetBracketsMat,
  );
  sc3bfGroup.add(targetBrackets);

  // Ultra-clean segmented horizontal SC3BF separation perimeter ring (R_sep bounding volume)
  const sepRingGeo = createSegmentedRingGeometry(R_SEP_M * 0.48, 36, 0.52, 0.055);
  sepRingGeo.rotateX(Math.PI * 0.5);
  const sepRingMat = new THREE.LineBasicMaterial({
    color: COLOR_NOMINAL_CYAN,
    transparent: true,
    opacity: 0.42,
    depthWrite: false,
  });
  const sepRing = new THREE.LineSegments(sepRingGeo, sepRingMat);
  sc3bfGroup.add(sepRing);

  // Subtle inner horizontal range-uncertainty ring that tightens as ActiveRange converges
  const sepInnerGeo = createSegmentedRingGeometry(1.0, 16, 0.42, 0.0);
  sepInnerGeo.rotateX(Math.PI * 0.5);
  const sepInnerMat = new THREE.LineBasicMaterial({
    color: COLOR_CAUTION_AMBER,
    transparent: true,
    opacity: 0.38,
    depthWrite: false,
  });
  const sepInnerRing = new THREE.LineSegments(sepInnerGeo, sepInnerMat);
  sc3bfGroup.add(sepInnerRing);

  // ---------------------------------------------------------------------------
  // 3. Crisp 3D Avoidance Trajectory Corridor Arc (<=12° Planned S-Swerve Maneuver)
  // ---------------------------------------------------------------------------
  const avoidGroup = new THREE.Group();
  avoidGroup.name = "avoidance_trajectory_corridor";
  group.add(avoidGroup);

  const AVOID_SEGS = 36;
  const AVOID_PTS = AVOID_SEGS + 1;
  const avoidCenterPos = new Float32Array(AVOID_PTS * 3);
  const avoidLeftPos = new Float32Array(AVOID_PTS * 3);
  const avoidRightPos = new Float32Array(AVOID_PTS * 3);
  const avoidRibbonPos = new Float32Array(AVOID_PTS * 2 * 3);
  const avoidRibbonUvs = new Float32Array(AVOID_PTS * 2 * 2);
  const avoidRibbonIndices: number[] = [];
  for (let i = 0; i < AVOID_SEGS; i++) {
    const a = i * 2;
    const b = a + 1;
    const c = a + 2;
    const d = a + 3;
    avoidRibbonIndices.push(a, b, c, b, d, c);
  }

  const avoidCenterGeo = new THREE.BufferGeometry();
  avoidCenterGeo.setAttribute("position", new THREE.BufferAttribute(avoidCenterPos, 3));
  const avoidCenterMat = new THREE.LineBasicMaterial({
    color: COLOR_EMERALD_LOCK,
    transparent: true,
    opacity: 0.95,
    depthWrite: false,
  });
  const avoidCenterLine = new THREE.Line(avoidCenterGeo, avoidCenterMat);
  avoidCenterLine.frustumCulled = false;
  avoidGroup.add(avoidCenterLine);

  const avoidRailMat = new THREE.LineBasicMaterial({
    color: COLOR_EMERALD_BRIGHT,
    transparent: true,
    opacity: 0.64,
    depthWrite: false,
  });
  const avoidLeftGeo = new THREE.BufferGeometry();
  avoidLeftGeo.setAttribute("position", new THREE.BufferAttribute(avoidLeftPos, 3));
  const avoidLeftLine = new THREE.Line(avoidLeftGeo, avoidRailMat);
  avoidLeftLine.frustumCulled = false;
  avoidGroup.add(avoidLeftLine);

  const avoidRightGeo = new THREE.BufferGeometry();
  avoidRightGeo.setAttribute("position", new THREE.BufferAttribute(avoidRightPos, 3));
  const avoidRightLine = new THREE.Line(avoidRightGeo, avoidRailMat);
  avoidRightLine.frustumCulled = false;
  avoidGroup.add(avoidRightLine);

  const avoidRibbonGeo = new THREE.BufferGeometry();
  avoidRibbonGeo.setAttribute("position", new THREE.BufferAttribute(avoidRibbonPos, 3));
  avoidRibbonGeo.setAttribute("uv", new THREE.BufferAttribute(avoidRibbonUvs, 2));
  avoidRibbonGeo.setIndex(avoidRibbonIndices);
  const avoidRibbonMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.NormalBlending,
    uniforms: {
      uColor: { value: new THREE.Color(COLOR_EMERALD_LOCK) },
      uTime: { value: 0.0 },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      uniform float uTime;
      varying vec2 vUv;
      void main() {
        float across = smoothstep(0.0, 0.14, vUv.x) * smoothstep(1.0, 0.86, vUv.x);
        float along = smoothstep(0.0, 0.07, vUv.y) * smoothstep(1.0, 0.80, vUv.y);
        float pulse = 0.82 + 0.18 * sin(vUv.y * 22.0 - uTime * 3.8);
        float alpha = across * along * 0.15 * pulse;
        gl_FragColor = vec4(uColor, alpha);
      }
    `,
  });
  const avoidRibbonMesh = new THREE.Mesh(avoidRibbonGeo, avoidRibbonMat);
  avoidRibbonMesh.frustumCulled = false;
  avoidGroup.add(avoidRibbonMesh);

  // 8 directional trajectory chevron ticks along the planned S-swerve arc
  const AVOID_CHEVRON_COUNT = 8;
  const avoidChevronPos = new Float32Array(AVOID_CHEVRON_COUNT * 4 * 3);
  const avoidChevronGeo = new THREE.BufferGeometry();
  avoidChevronGeo.setAttribute("position", new THREE.BufferAttribute(avoidChevronPos, 3));
  const avoidChevronMat = new THREE.LineBasicMaterial({
    color: COLOR_EMERALD_BRIGHT,
    transparent: true,
    opacity: 0.90,
    depthWrite: false,
  });
  const avoidChevrons = new THREE.LineSegments(avoidChevronGeo, avoidChevronMat);
  avoidChevrons.frustumCulled = false;
  avoidGroup.add(avoidChevrons);

  // ---------------------------------------------------------------------------
  // 4. XVS-ForcedLand 3D Wind-Warped Trochoid Highway Ribbon + Holographic Gates
  // ---------------------------------------------------------------------------
  const flGroup = new THREE.Group();
  group.add(flGroup);

  const MAX_FL_SEGS = 20;
  const ribbonPositions = new Float32Array((MAX_FL_SEGS + 1) * 2 * 3);
  const ribbonUvs = new Float32Array((MAX_FL_SEGS + 1) * 2 * 2);
  const ribbonIndices: number[] = [];
  for (let i = 0; i < MAX_FL_SEGS; i++) {
    const a = i * 2;
    const b = a + 1;
    const c = a + 2;
    const d = a + 3;
    ribbonIndices.push(a, b, c, b, d, c);
  }
  const ribbonGeo = new THREE.BufferGeometry();
  ribbonGeo.setAttribute("position", new THREE.BufferAttribute(ribbonPositions, 3));
  ribbonGeo.setAttribute("uv", new THREE.BufferAttribute(ribbonUvs, 2));
  ribbonGeo.setIndex(ribbonIndices);

  const ribbonMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uTime: { value: 0.0 },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float uTime;
      varying vec2 vUv;
      void main() {
        float edge = smoothstep(0.0, 0.12, vUv.x) * smoothstep(1.0, 0.88, vUv.x);
        float border = smoothstep(0.41, 0.495, abs(vUv.x - 0.5));
        float chevron = smoothstep(0.68, 0.95, sin((vUv.y * 18.0 - uTime * 3.4) * 3.14159 - abs(vUv.x - 0.5) * 9.0));
        vec3 baseCol = mix(vec3(0.05, 0.54, 0.40), vec3(0.76, 0.48, 0.06), vUv.y * 0.5);
        vec3 col = baseCol * (0.52 + 0.85 * border + 0.55 * chevron);
        float fadeStart = smoothstep(0.0, 0.08, vUv.y);
        float alpha = fadeStart * (0.10 * edge + 0.42 * border + 0.24 * chevron * edge) * (0.45 + 0.55 * (1.0 - vUv.y * 0.4));
        gl_FragColor = vec4(col, clamp(alpha, 0.0, 0.62));
      }
    `,
  });
  const ribbonMesh = new THREE.Mesh(ribbonGeo, ribbonMat);
  ribbonMesh.frustumCulled = false;
  flGroup.add(ribbonMesh);

  // 14 floating holographic rectangular chevron tunnel gates along the trochoid
  const gateCount = 14;
  const gateFrameGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(2.0, 2.0, 0.25));
  const gateFrameMat = new THREE.LineBasicMaterial({
    color: COLOR_EMERALD_LOCK,
    transparent: true,
    opacity: 0.65,
  });
  const gates: THREE.LineSegments[] = [];
  for (let i = 0; i < gateCount; i++) {
    const g = new THREE.LineSegments(gateFrameGeo, gateFrameMat);
    flGroup.add(g);
    gates.push(g);
  }

  // ---------------------------------------------------------------------------
  // 5. Optional 6-Cam Volumetric Sensor Dome (default OFF for uncluttered hero shot)
  //    Translucent Fresnel aerospace dome + 6 optical frustum pyramids + crisp segmented azimuth ring
  // ---------------------------------------------------------------------------
  const domeGroup = new THREE.Group();
  domeGroup.visible = false;
  group.add(domeGroup);

  const domeShellMat = createFresnelVolumeMaterial(COLOR_NOMINAL_CYAN, 0.042);
  const domeSphere = new THREE.Mesh(new THREE.SphereGeometry(24.5, 36, 24), domeShellMat);
  domeGroup.add(domeSphere);

  // Crisp segmented holographic azimuth horizon ring
  const domeEqGeo = createSegmentedRingGeometry(24.8, 36, 0.62, 0.05);
  domeEqGeo.rotateX(Math.PI * 0.5);
  const domeEqMat = new THREE.LineBasicMaterial({
    color: COLOR_NOMINAL_CYAN,
    transparent: true,
    opacity: 0.44,
    depthWrite: false,
  });
  const domeEqRing = new THREE.LineSegments(domeEqGeo, domeEqMat);
  domeGroup.add(domeEqRing);

  // 6 sleek non-overlapping truncated optical camera frustum pyramids radiating cleanly outside fuselage
  const camConeGeo = new THREE.CylinderGeometry(4.6, 0.75, 17.8, 4, 1, true);
  camConeGeo.rotateY(Math.PI * 0.25);
  camConeGeo.rotateX(-Math.PI * 0.5);
  camConeGeo.translate(0, 0, -14.8);
  const camConeMat = createFresnelVolumeMaterial(0x22d3ee, 0.055);
  const camEdgeGeo = new THREE.EdgesGeometry(camConeGeo, 12);
  const camEdgeMat = new THREE.LineBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.34,
    depthWrite: false,
  });
  for (let c = 0; c < 6; c++) {
    const camSector = new THREE.Group();
    camSector.rotation.y = (c * Math.PI) / 3;
    camSector.add(new THREE.Mesh(camConeGeo, camConeMat));
    camSector.add(new THREE.LineSegments(camEdgeGeo, camEdgeMat));
    domeGroup.add(camSector);
  }

  // ---------------------------------------------------------------------------
  // 6. GPS Spoof Quarantine Shield + Ghost Marker + XVS-Nav Visual Ground-Lock Beams
  // ---------------------------------------------------------------------------
  const spoofGroup = new THREE.Group();
  spoofGroup.visible = false;
  group.add(spoofGroup);

  const ghostMat = new THREE.LineBasicMaterial({
    color: COLOR_ALERT_CRIMSON,
    transparent: true,
    opacity: 0.85,
    depthWrite: false,
  });
  const ghostMarker = new THREE.LineSegments(
    createTacticalDiamondGeometry(11.0),
    ghostMat,
  );
  spoofGroup.add(ghostMarker);

  const ghostLinkGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(1, 0, 0),
  ]);
  const ghostLink = new THREE.Line(
    ghostLinkGeo,
    new THREE.LineBasicMaterial({ color: COLOR_ALERT_CRIMSON, transparent: true, opacity: 0.82 }),
  );
  spoofGroup.add(ghostLink);

  // Ownship-centered GPS Spoof Quarantine Shield (outer crimson GNSS reject ring + inner emerald XVS-Nav lock ring)
  const spoofShieldRig = new THREE.Group();
  const spoofOuterGeo = createSegmentedRingGeometry(20.0, 24, 0.52, 0.09);
  spoofOuterGeo.rotateX(Math.PI * 0.5);
  const spoofOuterRing = new THREE.LineSegments(
    spoofOuterGeo,
    new THREE.LineBasicMaterial({
      color: COLOR_ALERT_CRIMSON,
      transparent: true,
      opacity: 0.88,
      depthWrite: false,
    }),
  );
  spoofShieldRig.add(spoofOuterRing);

  const spoofInnerGeo = createSegmentedRingGeometry(15.2, 32, 0.68, 0.05);
  spoofInnerGeo.rotateX(Math.PI * 0.5);
  const spoofInnerRing = new THREE.LineSegments(
    spoofInnerGeo,
    new THREE.LineBasicMaterial({
      color: COLOR_EMERALD_BRIGHT,
      transparent: true,
      opacity: 0.90,
      depthWrite: false,
    }),
  );
  spoofShieldRig.add(spoofInnerRing);
  spoofGroup.add(spoofShieldRig);

  // 4 glowing emerald cross-view ground-lock rays from ownship to KSFO landmarks + ground lock reticles
  const navBeamMat = new THREE.LineBasicMaterial({
    color: COLOR_EMERALD_LOCK,
    transparent: true,
    opacity: 0.68,
    depthWrite: false,
  });
  const navTargets = [
    new THREE.Vector3(0, 2, 0), // 28L threshold
    new THREE.Vector3(-1200, 40, -520), // ATC Tower
    new THREE.Vector3(-900, 6, -900), // Terminal Complex
    new THREE.Vector3(-2200, 5, -1080), // Runway 28L midpoint
  ];
  const navMarkerGeo = createTacticalDiamondGeometry(18.0);
  const navBeams = navTargets.map((tgt) => {
    const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), tgt.clone()]);
    const line = new THREE.Line(geo, navBeamMat);
    spoofGroup.add(line);
    const marker = new THREE.LineSegments(navMarkerGeo, navBeamMat);
    marker.position.copy(tgt);
    spoofGroup.add(marker);
    return { line, target: tgt };
  });

  // Reusable scratch vectors (zero per-frame GC allocations)
  const vOwn = new THREE.Vector3();
  const vIntr = new THREE.Vector3();
  const vDir = new THREE.Vector3();
  const vUp = new THREE.Vector3(0, 1, 0);
  const vRight = new THREE.Vector3();
  const vFwd = new THREE.Vector3();
  const vGatePos = new THREE.Vector3();
  const vGateNext = new THREE.Vector3();
  const mLook = new THREE.Matrix4();

  return {
    group,
    setSensorDomeVisible(visible: boolean): void {
      domeGroup.visible = visible;
    },
    isSensorDomeVisible(): boolean {
      return domeGroup.visible;
    },
    update(snap: StudioSimSnapshot, simTimeS: number): void {
      vOwn.set(snap.ownship.pos.x, snap.ownship.pos.y, snap.ownship.pos.z);
      vIntr.set(snap.intruder.pos.x, snap.intruder.pos.y, snap.intruder.pos.z);

      // Update 6-cam sensor dome position/orientation & Fresnel pulse
      domeGroup.position.copy(vOwn);
      domeGroup.rotation.set(
        snap.ownship.pitchRad,
        -snap.ownship.headingRad,
        -snap.ownship.rollRad,
        "YXZ",
      );
      if (domeGroup.visible) {
        domeShellMat.uniforms.uTime.value = simTimeS;
        camConeMat.uniforms.uTime.value = simTimeS;
      }

      // 1, 2 & 3. ActiveRange FOV Cone, Tactical Target Designator & <=12° Avoidance Corridor Arc
      const showAr = snap.intruderVisible;
      arGroup.visible = showAr;
      sc3bfGroup.visible = showAr;
      avoidGroup.visible = showAr;

      if (showAr) {
        vDir.subVectors(vIntr, vOwn);
        const distM = Math.max(40.0, vDir.length());
        arGroup.position.copy(vOwn);
        arGroup.lookAt(vIntr);

        const ar = snap.activeRange;
        const evade = ar.mode === ActiveRangeModeTs.ImmediateEvade || !snap.sc3bfSafe;
        const locked = !evade && (ar.mode === ActiveRangeModeTs.Locked || ar.sigmaRM <= 95.0);
        const relSigma = Math.min(0.62, Math.max(0.03, ar.sigmaRM / Math.max(ar.rHatM, 120.0)));

        // Threat-driven aerospace avionics color coding: Alert Crimson / Caution Amber / Nominal Cyan
        const threatHex = evade
          ? COLOR_ALERT_CRIMSON
          : locked
            ? COLOR_NOMINAL_CYAN
            : COLOR_CAUTION_AMBER;

        // 1a. Sleek translucent FOV cone (normal-blended opacity 0.075–0.095 + crisp edge hairlines)
        const startOffsetM = 24.0;
        const coneLen = Math.max(
          80.0,
          Math.min(distM - startOffsetM + 12.0, ar.rHatM + ar.sigmaRM * 0.30 - startOffsetM),
        );
        const baseRadius = Math.max(14.0, distM * (0.030 + relSigma * 0.095));
        coneMesh.position.set(0, 0, startOffsetM);
        coneMesh.scale.set(baseRadius, baseRadius, coneLen);
        coneMat.uniforms.uTime.value = simTimeS;
        coneMat.uniforms.uColor.value.setHex(threatHex);
        coneMat.uniforms.uOpacity.value = evade ? 0.098 : locked ? 0.074 : 0.088;

        coneEdges.position.set(0, 0, startOffsetM);
        coneEdges.scale.set(baseRadius, baseRadius, coneLen);
        coneEdgesMat.color.setHex(threatHex);
        coneEdgesMat.opacity = evade ? 0.44 : locked ? 0.32 : 0.38;

        // 2a. Precision 3D Tactical Target Diamond + 8-Corner Brackets framing CBDR-1
        const bracketScale = Math.max(15.0, Math.min(34.0, 14.5 + distM * 0.015));
        sc3bfGroup.position.copy(vIntr);
        targetDiamond.scale.setScalar(bracketScale);
        targetDiamond.rotation.y = simTimeS * 0.28;
        targetDiamondMat.color.setHex(threatHex);

        targetInnerDiamond.scale.setScalar(bracketScale);
        targetInnerDiamond.rotation.y = simTimeS * 0.28;
        targetInnerDiamondMat.color.setHex(
          evade ? 0xb91c1c : locked ? COLOR_DEEP_CYAN : 0xb45309,
        );

        targetBrackets.scale.setScalar(bracketScale * 0.86);
        targetBrackets.lookAt(vOwn);
        targetBracketsMat.color.setHex(threatHex);

        // 1b. Thin LOS centerline terminating cleanly outside target diamond
        const losEnd = Math.max(startOffsetM + 12.0, distM - bracketScale * 1.45);
        const losArr = losLine.geometry.attributes.position.array as Float32Array;
        losArr[2] = 12.0;
        losArr[5] = losEnd;
        losLine.geometry.attributes.position.needsUpdate = true;
        losLineMat.color.setHex(threatHex);

        // 1c. Subtle thin segmented range-uncertainty rings along cone (zero thick torus discs)
        const zHat = Math.max(150.0, Math.min(distM * 0.96, ar.rHatM));
        const fracHat = Math.max(0.14, (zHat - startOffsetM) / Math.max(coneLen, 1.0));
        const radHat = Math.max(12.0, fracHat * baseRadius);
        rangeHatRing.position.set(0, 0, zHat);
        rangeHatRing.scale.set(radHat, radHat, 1.0);
        rangeHatRing.rotation.z = simTimeS * 0.14;
        rangeHatRingMat.color.setHex(threatHex);
        rangeHatRingMat.opacity = locked ? 0.40 : 0.46;

        const zLower = Math.max(125.0, Math.min(zHat - 25.0, ar.rSafeLowerM));
        const fracLower = Math.max(0.12, (zLower - startOffsetM) / Math.max(coneLen, 1.0));
        const radLower = Math.max(10.0, fracLower * baseRadius);
        rangeLowerRing.position.set(0, 0, zLower);
        rangeLowerRing.scale.set(radLower, radLower, 1.0);
        rangeLowerRingMat.color.setHex(evade ? COLOR_ALERT_CRIMSON : COLOR_CAUTION_AMBER);
        rangeLowerRingMat.opacity = 0.38;

        // 2b. Subtle thin horizontal SC3BF separation perimeter ring + converging uncertainty ring
        sepRing.rotation.y = -simTimeS * 0.12;
        sepRingMat.color.setHex(snap.sc3bfSafe ? COLOR_NOMINAL_CYAN : COLOR_ALERT_CRIMSON);
        sepRingMat.opacity = 0.42;

        const innerUncertaintyR = Math.max(
          bracketScale * 1.35,
          Math.min(R_SEP_M * 0.44, bracketScale * 1.25 + relSigma * 95.0),
        );
        sepInnerRing.scale.setScalar(innerUncertaintyR);
        sepInnerRing.rotation.y = simTimeS * 0.18;
        sepInnerMat.color.setHex(threatHex);
        sepInnerMat.opacity = locked ? 0.36 : 0.44;

        // 3. Crisp 3D Avoidance Trajectory Corridor Arc (<=12° Planned S-Swerve Maneuver)
        vFwd
          .set(Math.sin(snap.ownship.headingRad), 0, -Math.cos(snap.ownship.headingRad))
          .normalize();
        vRight.crossVectors(vFwd, vUp).normalize();

        // FAR 91.113 right-turn preference unless ActiveRange explicitly commands left bank
        const swerveSign =
          Math.abs(ar.cmdBankRad) > 0.015
            ? Math.sign(ar.cmdBankRad)
            : Math.abs(snap.ownship.rollRad) > 0.015
              ? Math.sign(snap.ownship.rollRad)
              : 1.0;

        const arcStartM = 20.0;
        const arcLenM = Math.max(260.0, Math.min(720.0, distM * 0.80));
        const swerveAngleDeg = evade ? 11.8 : locked ? 8.8 : 11.2; // Strictly <= 12°
        const peakLatM = (Math.tan((swerveAngleDeg * Math.PI) / 180.0) * arcLenM) / Math.PI;
        const glideSlope = snap.ownship.vel.y / Math.max(25.0, snap.ownship.tasMps);
        const halfCorridorM = 8.5;

        for (let i = 0; i <= AVOID_SEGS; i++) {
          const t = i / AVOID_SEGS;
          const s = arcStartM + t * arcLenM;
          // Raised-cosine C1 S-swerve: tangent at t=0, peak lateral excursion at t=0.5, rejoins at t=1
          const lat = swerveSign * peakLatM * 0.5 * (1.0 - Math.cos(2.0 * Math.PI * t));
          const cx = vOwn.x + vFwd.x * s + vRight.x * lat;
          const cy = Math.max(12.0, vOwn.y + glideSlope * s - 1.2);
          const cz = vOwn.z + vFwd.z * s + vRight.z * lat;

          // Local lateral tangent normal for crisp corridor rails
          const dLatDs = swerveSign * peakLatM * (Math.PI / arcLenM) * Math.sin(2.0 * Math.PI * t);
          const tx = vFwd.x + vRight.x * dLatDs;
          const tz = vFwd.z + vRight.z * dLatDs;
          const tInv = 1.0 / Math.max(1e-4, Math.hypot(tx, tz));
          const nx = -tz * tInv;
          const nz = tx * tInv;

          const pIdx = i * 3;
          avoidCenterPos[pIdx + 0] = cx;
          avoidCenterPos[pIdx + 1] = cy;
          avoidCenterPos[pIdx + 2] = cz;

          const lx = cx - nx * halfCorridorM;
          const lz = cz - nz * halfCorridorM;
          const rx = cx + nx * halfCorridorM;
          const rz = cz + nz * halfCorridorM;

          avoidLeftPos[pIdx + 0] = lx;
          avoidLeftPos[pIdx + 1] = cy;
          avoidLeftPos[pIdx + 2] = lz;

          avoidRightPos[pIdx + 0] = rx;
          avoidRightPos[pIdx + 1] = cy;
          avoidRightPos[pIdx + 2] = rz;

          const rIdx = i * 6;
          avoidRibbonPos[rIdx + 0] = lx;
          avoidRibbonPos[rIdx + 1] = cy;
          avoidRibbonPos[rIdx + 2] = lz;
          avoidRibbonPos[rIdx + 3] = rx;
          avoidRibbonPos[rIdx + 4] = cy;
          avoidRibbonPos[rIdx + 5] = rz;

          avoidRibbonUvs[i * 4 + 0] = 0.0;
          avoidRibbonUvs[i * 4 + 1] = t;
          avoidRibbonUvs[i * 4 + 2] = 1.0;
          avoidRibbonUvs[i * 4 + 3] = t;
        }
        avoidCenterGeo.attributes.position.needsUpdate = true;
        avoidLeftGeo.attributes.position.needsUpdate = true;
        avoidRightGeo.attributes.position.needsUpdate = true;
        avoidRibbonGeo.attributes.position.needsUpdate = true;
        avoidRibbonGeo.attributes.uv.needsUpdate = true;

        // Animated forward V-chevrons sliding cleanly along the planned avoidance arc
        const phase = (simTimeS * 0.26) % 1.0;
        const chevWingM = 6.2;
        const chevBackM = 9.5;
        for (let c = 0; c < AVOID_CHEVRON_COUNT; c++) {
          const t = ((c + phase) / AVOID_CHEVRON_COUNT) * 0.88 + 0.06;
          const s = arcStartM + t * arcLenM;
          const lat = swerveSign * peakLatM * 0.5 * (1.0 - Math.cos(2.0 * Math.PI * t));
          const cx = vOwn.x + vFwd.x * s + vRight.x * lat;
          const cy = Math.max(12.0, vOwn.y + glideSlope * s - 1.0);
          const cz = vOwn.z + vFwd.z * s + vRight.z * lat;

          const dLatDs = swerveSign * peakLatM * (Math.PI / arcLenM) * Math.sin(2.0 * Math.PI * t);
          const txRaw = vFwd.x + vRight.x * dLatDs;
          const tzRaw = vFwd.z + vRight.z * dLatDs;
          const tInv = 1.0 / Math.max(1e-4, Math.hypot(txRaw, tzRaw));
          const tx = txRaw * tInv;
          const tz = tzRaw * tInv;
          const nx = -tz;
          const nz = tx;

          const bx = cx - tx * chevBackM;
          const bz = cz - tz * chevBackM;
          const cIdx = c * 12;
          // Left wing -> apex
          avoidChevronPos[cIdx + 0] = bx - nx * chevWingM;
          avoidChevronPos[cIdx + 1] = cy;
          avoidChevronPos[cIdx + 2] = bz - nz * chevWingM;
          avoidChevronPos[cIdx + 3] = cx;
          avoidChevronPos[cIdx + 4] = cy;
          avoidChevronPos[cIdx + 5] = cz;
          // Right wing -> apex
          avoidChevronPos[cIdx + 6] = bx + nx * chevWingM;
          avoidChevronPos[cIdx + 7] = cy;
          avoidChevronPos[cIdx + 8] = bz + nz * chevWingM;
          avoidChevronPos[cIdx + 9] = cx;
          avoidChevronPos[cIdx + 10] = cy;
          avoidChevronPos[cIdx + 11] = cz;
        }
        avoidChevronGeo.attributes.position.needsUpdate = true;

        const avoidHex = evade
          ? COLOR_ALERT_CRIMSON
          : locked
            ? COLOR_EMERALD_LOCK
            : COLOR_CAUTION_AMBER;
        const avoidAccentHex = evade
          ? COLOR_ALERT_CRIMSON
          : locked
            ? COLOR_EMERALD_BRIGHT
            : COLOR_CAUTION_AMBER;
        avoidCenterMat.color.setHex(avoidHex);
        avoidRailMat.color.setHex(avoidAccentHex);
        avoidChevronMat.color.setHex(avoidAccentHex);
        avoidRibbonMat.uniforms.uColor.value.setHex(avoidHex);
        avoidRibbonMat.uniforms.uTime.value = simTimeS;
      }

      // 4. ForcedLand 3D Wind-Warped Trochoid Highway Ribbon (visible in forcedland & thermal_fog)
      const showFl =
        (snap.scenario === "forcedland" || snap.scenario === "thermal_fog") &&
        snap.forcedLandWaypoints.length >= 2;
      flGroup.visible = showFl;
      if (showFl) {
        ribbonMat.uniforms.uTime.value = simTimeS;
        const wps = snap.forcedLandWaypoints;
        const count = Math.min(MAX_FL_SEGS + 1, wps.length);
        for (let i = 0; i < count; i++) {
          const wp = wps[i];
          const next = wps[Math.min(count - 1, i + 1)];
          const prev = wps[Math.max(0, i - 1)];
          vFwd
            .set(next.pos.x - prev.pos.x, next.pos.y - prev.pos.y, next.pos.z - prev.pos.z)
            .normalize();
          vRight.crossVectors(vFwd, vUp).normalize();

          const hw = wp.halfWidthM * 0.38;
          const idx = i * 6;
          ribbonPositions[idx + 0] = wp.pos.x - vRight.x * hw;
          ribbonPositions[idx + 1] = wp.pos.y - 6.0;
          ribbonPositions[idx + 2] = wp.pos.z - vRight.z * hw;
          ribbonPositions[idx + 3] = wp.pos.x + vRight.x * hw;
          ribbonPositions[idx + 4] = wp.pos.y - 6.0;
          ribbonPositions[idx + 5] = wp.pos.z + vRight.z * hw;

          const vCoord = i / Math.max(1, count - 1);
          ribbonUvs[i * 4 + 0] = 0.0;
          ribbonUvs[i * 4 + 1] = vCoord;
          ribbonUvs[i * 4 + 2] = 1.0;
          ribbonUvs[i * 4 + 3] = vCoord;
        }
        ribbonGeo.attributes.position.needsUpdate = true;
        ribbonGeo.attributes.uv.needsUpdate = true;
        ribbonGeo.setDrawRange(0, (count - 1) * 6);

        // Place holographic tunnel gates along trochoid
        for (let g = 0; g < gateCount; g++) {
          const wpIdx = Math.min(
            count - 2,
            Math.max(1, Math.floor(((g + 1) / (gateCount + 1)) * (count - 1))),
          );
          const wp = wps[wpIdx];
          const wpNext = wps[Math.min(count - 1, wpIdx + 1)];
          const gate = gates[g];
          gate.position.set(wp.pos.x, wp.pos.y, wp.pos.z);
          vGatePos.set(wp.pos.x, wp.pos.y, wp.pos.z);
          vGateNext.set(wpNext.pos.x, wpNext.pos.y, wpNext.pos.z);
          mLook.lookAt(vGatePos, vGateNext, vUp);
          gate.quaternion.setFromRotationMatrix(mLook);
          gate.scale.set(wp.halfWidthM * 0.52, wp.halfHeightM * 0.56, 1.0);
        }
        gateFrameMat.color.setHex(snap.scenario === "thermal_fog" ? 0xa78bfa : COLOR_EMERALD_LOCK);
      }

      // 6. GPS Spoof Quarantine Shield + Ghost Marker + XVS-Nav Visual Ground-Lock Rays
      spoofGroup.visible = snap.spoofAlarm;
      if (snap.spoofAlarm) {
        const gx = vOwn.x + snap.spoofGhostOffset.x;
        const gy = vOwn.y + snap.spoofGhostOffset.y;
        const gz = vOwn.z + snap.spoofGhostOffset.z;
        ghostMarker.position.set(gx, gy, gz);
        ghostMarker.rotation.y = simTimeS * 1.1;

        spoofShieldRig.position.copy(vOwn);
        spoofOuterRing.rotation.y = -simTimeS * 0.35;
        spoofInnerRing.rotation.y = simTimeS * 0.24;

        const gArr = ghostLink.geometry.attributes.position.array as Float32Array;
        gArr[0] = vOwn.x;
        gArr[1] = vOwn.y;
        gArr[2] = vOwn.z;
        gArr[3] = gx;
        gArr[4] = gy;
        gArr[5] = gz;
        ghostLink.geometry.attributes.position.needsUpdate = true;

        for (const nb of navBeams) {
          const arr = nb.line.geometry.attributes.position.array as Float32Array;
          arr[0] = vOwn.x;
          arr[1] = vOwn.y - 1.5;
          arr[2] = vOwn.z;
          arr[3] = nb.target.x;
          arr[4] = nb.target.y;
          arr[5] = nb.target.z;
          nb.line.geometry.attributes.position.needsUpdate = true;
        }
      }
    },
  };
}
