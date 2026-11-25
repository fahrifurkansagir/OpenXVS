/**
 * Luminous 3D Mathematical Safety Manifolds (`OpenXVS Studio v4.0`).
 *
 * Visually renders the formal aerospace mathematics of OpenXVS v3.0/v4.0 in glowing
 * Anduril Lattice / Tesla FSD holographic clarity:
 *   1. `XVS-ActiveRange` Collapsing Volumetric CRLB / Conformal Lock Cone:
 *      Wide glowing amber-coral frustum during `ACTIVE_SWERVE` that visibly collapses
 *      into a tight glowing emerald-cyan lock sphere (`±25m KİLİTLENDİ`) as range locks!
 *   2. `SC3BF` Shifted Collision-Cone Geodesic Shield (`R_sep = 152.4 m`) around `CBDR-1`.
 *   3. `XVS-ForcedLand` 3D Wind-Warped Trochoid Highway-in-the-Sky Ribbon + Chevron Gates.
 *   4. Optional `6-Cam` Volumetric Optical Sensor Dome around ownship.
 *   5. `GPS Spoof` Ghost Trajectory Marker & `XVS-Nav` Cross-View Ground Lock Rays.
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
        float scan = 0.85 + 0.15 * sin(vUv.y * 28.0 - uTime * 3.8);
        float alpha = clamp((0.14 + 0.86 * fresnel) * uOpacity * scan, 0.0, 0.92);
        gl_FragColor = vec4(uColor * (1.15 + 0.55 * fresnel), alpha);
      }
    `,
  });
}

export function createSafetyManifoldsRig(): SafetyManifoldsRig {
  const group = new THREE.Group();
  group.name = "openxvs_safety_manifolds";

  // ---------------------------------------------------------------------------
  // 1. XVS-ActiveRange Collapsing Volumetric Lock Cone + Range Rings + Lock Sphere
  // ---------------------------------------------------------------------------
  const arGroup = new THREE.Group();
  group.add(arGroup);

  // Unit frustum aligned along +Z from z=0 (ownship apex) to z=1 (base at rHat+sigma)
  const coneGeo = new THREE.CylinderGeometry(1.0, 0.04, 1.0, 40, 1, true);
  coneGeo.rotateX(Math.PI * 0.5);
  coneGeo.translate(0, 0, 0.5);
  const coneMat = createFresnelVolumeMaterial(0xf59e0b, 0.32);
  const coneMesh = new THREE.Mesh(coneGeo, coneMat);
  arGroup.add(coneMesh);

  // Glowing LOS centerline laser from ownship to intruder
  const losLineGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, 1),
  ]);
  const losLineMat = new THREE.LineBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.72,
  });
  const losLine = new THREE.Line(losLineGeo, losLineMat);
  arGroup.add(losLine);

  // 3 concentric glowing uncertainty range rings (rSafeLower, rHat, rUpper)
  const ringGeo = new THREE.TorusGeometry(1.0, 0.018, 8, 48);
  const ringMats = [
    new THREE.MeshBasicMaterial({ color: 0xf43f5e, transparent: true, opacity: 0.72 }),
    new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.92 }),
    new THREE.MeshBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.55 }),
  ];
  const rangeRings = ringMats.map((m) => {
    const mesh = new THREE.Mesh(ringGeo, m);
    arGroup.add(mesh);
    return mesh;
  });

  // Collapsing emerald lock sphere that wraps tightly around CBDR-1 once locked
  const lockSphereMat = createFresnelVolumeMaterial(0x10b981, 0.45);
  const lockSphere = new THREE.Mesh(new THREE.SphereGeometry(1.0, 28, 20), lockSphereMat);
  arGroup.add(lockSphere);

  // ---------------------------------------------------------------------------
  // 2. SC3BF Shifted Collision-Cone Shield around CBDR-1 (R_sep = 152.4 m)
  // ---------------------------------------------------------------------------
  const sc3bfGroup = new THREE.Group();
  group.add(sc3bfGroup);

  const shieldShellMat = createFresnelVolumeMaterial(0x22d3ee, 0.24);
  const shieldShell = new THREE.Mesh(new THREE.IcosahedronGeometry(R_SEP_M * 0.42, 2), shieldShellMat);
  sc3bfGroup.add(shieldShell);

  const shieldWireMat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    wireframe: true,
    transparent: true,
    opacity: 0.22,
  });
  const shieldWire = new THREE.Mesh(new THREE.IcosahedronGeometry(R_SEP_M * 0.425, 1), shieldWireMat);
  sc3bfGroup.add(shieldWire);

  const eqRingMat = new THREE.MeshBasicMaterial({
    color: 0x34d399,
    transparent: true,
    opacity: 0.78,
  });
  const eqRing = new THREE.Mesh(new THREE.TorusGeometry(R_SEP_M * 0.48, 0.85, 8, 56), eqRingMat);
  eqRing.rotation.x = Math.PI * 0.5;
  sc3bfGroup.add(eqRing);

  // ---------------------------------------------------------------------------
  // 3. XVS-ForcedLand 3D Wind-Warped Trochoid Highway Ribbon + Holographic Gates
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
    color: 0x34d399,
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
  // 4. Optional 6-Cam Volumetric Sensor Dome (default OFF for uncluttered hero shot)
  // ---------------------------------------------------------------------------
  const domeGroup = new THREE.Group();
  domeGroup.visible = false;
  group.add(domeGroup);

  const domeSphere = new THREE.Mesh(
    new THREE.SphereGeometry(28.0, 24, 14),
    new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      wireframe: true,
      transparent: true,
      opacity: 0.14,
    }),
  );
  domeGroup.add(domeSphere);

  // 6 optical camera frustum cones arranged around horizon
  const camConeGeo = new THREE.ConeGeometry(9.5, 28.0, 4, 1, true);
  camConeGeo.rotateX(-Math.PI * 0.5);
  camConeGeo.translate(0, 0, -14.0);
  const camConeMat = new THREE.MeshBasicMaterial({
    color: 0x22d3ee,
    wireframe: true,
    transparent: true,
    opacity: 0.18,
  });
  for (let c = 0; c < 6; c++) {
    const cone = new THREE.Mesh(camConeGeo, camConeMat);
    cone.rotation.y = (c * Math.PI) / 3;
    domeGroup.add(cone);
  }

  // ---------------------------------------------------------------------------
  // 5. GPS Spoof Ghost Marker + XVS-Nav Visual Ground-Lock Beams
  // ---------------------------------------------------------------------------
  const spoofGroup = new THREE.Group();
  spoofGroup.visible = false;
  group.add(spoofGroup);

  const ghostMat = new THREE.MeshBasicMaterial({
    color: 0xf43f5e,
    wireframe: true,
    transparent: true,
    opacity: 0.75,
  });
  const ghostMarker = new THREE.Mesh(new THREE.OctahedronGeometry(12.0, 1), ghostMat);
  spoofGroup.add(ghostMarker);

  const ghostLinkGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(1, 0, 0),
  ]);
  const ghostLink = new THREE.Line(
    ghostLinkGeo,
    new THREE.LineBasicMaterial({ color: 0xf43f5e, transparent: true, opacity: 0.85 }),
  );
  spoofGroup.add(ghostLink);

  // 4 glowing cyan cross-view ground-lock rays from ownship to KSFO landmarks
  const navBeamMat = new THREE.LineBasicMaterial({
    color: 0x34d399,
    transparent: true,
    opacity: 0.68,
  });
  const navTargets = [
    new THREE.Vector3(0, 2, 0), // 28L threshold
    new THREE.Vector3(-1200, 40, -520), // ATC Tower
    new THREE.Vector3(-900, 6, -900), // Terminal Complex
    new THREE.Vector3(-2200, 5, -1080), // Runway 28L midpoint
  ];
  const navBeams = navTargets.map((tgt) => {
    const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), tgt.clone()]);
    const line = new THREE.Line(geo, navBeamMat);
    spoofGroup.add(line);
    return { line, target: tgt };
  });

  // Reusable vectors
  const vOwn = new THREE.Vector3();
  const vIntr = new THREE.Vector3();
  const vDir = new THREE.Vector3();
  const vUp = new THREE.Vector3(0, 1, 0);
  const vRight = new THREE.Vector3();
  const vFwd = new THREE.Vector3();
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

      // Update 6-cam sensor dome position/orientation
      domeGroup.position.copy(vOwn);
      domeGroup.rotation.set(snap.ownship.pitchRad, -snap.ownship.headingRad, -snap.ownship.rollRad, "YXZ");

      // 1 & 2. ActiveRange Lock Cone + SC3BF Shield (visible in CBDR & GPS Spoof)
      const showAr = snap.intruderVisible;
      arGroup.visible = showAr;
      sc3bfGroup.visible = showAr;

      if (showAr) {
        vDir.subVectors(vIntr, vOwn);
        const distM = Math.max(40.0, vDir.length());
        arGroup.position.copy(vOwn);
        arGroup.lookAt(vIntr);

        const ar = snap.activeRange;
        const locked = ar.mode === ActiveRangeModeTs.Locked || ar.sigmaRM <= 95.0;
        const relSigma = Math.min(0.65, Math.max(0.03, ar.sigmaRM / Math.max(ar.rHatM, 120.0)));

        // Start cone cleanly 22 m ahead of nose radome so it never overlaps/tints the fuselage!
        const startOffsetM = 22.0;
        const coneLen = Math.max(80.0, Math.min(distM - startOffsetM + 18.0, ar.rHatM + ar.sigmaRM * 0.35 - startOffsetM));
        const baseRadius = Math.max(16.0, distM * (0.036 + relSigma * 0.13));
        coneMesh.position.set(0, 0, startOffsetM);
        coneMesh.scale.set(baseRadius, baseRadius, coneLen);
        coneMat.uniforms.uTime.value = simTimeS;
        coneMat.uniforms.uColor.value.setHex(
          ar.mode === ActiveRangeModeTs.ImmediateEvade
            ? 0xf43f5e
            : locked
              ? 0x10b981
              : 0xf59e0b,
        );
        coneMat.uniforms.uOpacity.value = locked ? 0.16 : 0.22;

        // Update LOS laser line length (starts cleanly at nose radome z = 8 m)
        const losArr = losLine.geometry.attributes.position.array as Float32Array;
        losArr[2] = 8.0;
        losArr[5] = distM;
        losLine.geometry.attributes.position.needsUpdate = true;
        losLineMat.color.setHex(locked ? 0x34d399 : 0x38bdf8);

        // Position 3 concentric uncertainty rings along the cone (keep strictly > 120 m ahead of ownship)
        const zRings = [
          Math.max(140.0, Math.min(distM * 0.9, ar.rSafeLowerM)),
          Math.max(160.0, Math.min(distM, ar.rHatM)),
          Math.max(180.0, Math.min(distM * 1.03, ar.rHatM + ar.sigmaRM * 0.25)),
        ];
        for (let i = 0; i < 3; i++) {
          const zr = zRings[i];
          const frac = Math.max(0.12, (zr - startOffsetM) / Math.max(coneLen, 1.0));
          const rad = Math.max(10.0, frac * baseRadius);
          rangeRings[i].position.set(0, 0, zr);
          rangeRings[i].scale.set(rad, rad, 1.0);
        }

        // Tight collapsing emerald lock sphere at intruder range
        const sphereR = Math.max(18.0, Math.min(72.0, 18.0 + relSigma * 120.0));
        lockSphere.position.set(0, 0, distM);
        lockSphere.scale.setScalar(sphereR);
        lockSphereMat.uniforms.uTime.value = simTimeS;
        lockSphereMat.uniforms.uColor.value.setHex(locked ? 0x10b981 : 0xf59e0b);
        lockSphereMat.uniforms.uOpacity.value = locked ? 0.34 : 0.22;

        // SC3BF geodesic collision-cone barrier around CBDR-1
        sc3bfGroup.position.copy(vIntr);
        shieldShell.rotation.y = simTimeS * 0.32;
        shieldWire.rotation.y = -simTimeS * 0.22;
        shieldShellMat.uniforms.uTime.value = simTimeS;
        const scColor = snap.sc3bfSafe ? 0x22d3ee : 0xf43f5e;
        shieldShellMat.uniforms.uColor.value.setHex(scColor);
        shieldWireMat.color.setHex(scColor);
        eqRingMat.color.setHex(snap.sc3bfSafe ? 0x34d399 : 0xf43f5e);
      }

      // 3. ForcedLand 3D Wind-Warped Trochoid Highway Ribbon (visible in forcedland & thermal_fog)
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
          vFwd.set(next.pos.x - prev.pos.x, next.pos.y - prev.pos.y, next.pos.z - prev.pos.z).normalize();
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
          const wpIdx = Math.min(count - 2, Math.max(1, Math.floor(((g + 1) / (gateCount + 1)) * (count - 1))));
          const wp = wps[wpIdx];
          const wpNext = wps[Math.min(count - 1, wpIdx + 1)];
          const gate = gates[g];
          gate.position.set(wp.pos.x, wp.pos.y, wp.pos.z);
          mLook.lookAt(
            new THREE.Vector3(wp.pos.x, wp.pos.y, wp.pos.z),
            new THREE.Vector3(wpNext.pos.x, wpNext.pos.y, wpNext.pos.z),
            vUp,
          );
          gate.quaternion.setFromRotationMatrix(mLook);
          gate.scale.set(wp.halfWidthM * 0.52, wp.halfHeightM * 0.56, 1.0);
        }
        gateFrameMat.color.setHex(snap.scenario === "thermal_fog" ? 0xa78bfa : 0x34d399);
      }

      // 5. GPS Spoof Ghost Marker + XVS-Nav Visual Ground-Lock Rays
      spoofGroup.visible = snap.spoofAlarm;
      if (snap.spoofAlarm) {
        const gx = vOwn.x + snap.spoofGhostOffset.x;
        const gy = vOwn.y + snap.spoofGhostOffset.y;
        const gz = vOwn.z + snap.spoofGhostOffset.z;
        ghostMarker.position.set(gx, gy, gz);
        ghostMarker.rotation.y = simTimeS * 1.4;
        ghostMarker.rotation.x = simTimeS * 0.8;

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
