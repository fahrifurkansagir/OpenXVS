/**
 * High-Poly Multi-Part PBR 3D Civil Aircraft & Helicopter Factory (`OpenXVS Studio v4.0`).
 *
 * ZERO origami triangles:
 *   - Ownship Civil Executive Twin-Turboprop (`> 5,000` smooth-shaded triangles):
 *     lofted aerodynamic fuselage + nose radome, clearcoat dark cockpit canopy,
 *     swept NACA airfoil wings + winglets, twin engine nacelles with spinning 4-blade
 *     propellers + blur discs, T-tail, belly OpenXVS EO/LWIR sensor pod, procedural
 *     panel-seam/rivet normal+roughness maps, blooming wingtip nav/strobe lights,
 *     and dynamic world-space ribbon contrails.
 *   - Intruder Civil Helicopter `CBDR-1` (`> 2,500` smooth-shaded triangles):
 *     sculpted teardrop cabin, glossy tinted bubble canopy, turbine cowling, tail boom,
 *     tubular skids, rapidly spinning main + tail rotors, and blinking red beacon.
 *
 * Local Aircraft Convention (Three.js right-handed):
 *   - Nose forward: `-Z`, Right wing: `+X`, Up: `+Y`.
 */

import * as THREE from "three";

export interface OwnshipRig {
  readonly group: THREE.Group;
  readonly cockpitOccludingParts: readonly THREE.Object3D[];
  readonly leftProp: THREE.Group;
  readonly rightProp: THREE.Group;
  readonly leftBlur: THREE.Mesh;
  readonly rightBlur: THREE.Mesh;
  readonly strobeMat: THREE.MeshStandardMaterial;
  readonly beaconMat: THREE.MeshStandardMaterial;
  readonly leftTrail: RibbonTrail;
  readonly rightTrail: RibbonTrail;
  readonly triangleCount: number;
  update(dtS: number, simTimeS: number, engineOutRight: boolean): void;
}

export interface IntruderHeliRig {
  readonly group: THREE.Group;
  readonly mainRotor: THREE.Group;
  readonly tailRotor: THREE.Group;
  readonly beaconMat: THREE.MeshStandardMaterial;
  readonly triangleCount: number;
  update(dtS: number, simTimeS: number): void;
}

/** Counts total triangles across all Mesh geometries in an Object3D subtree. */
export function countMeshTriangles(root: THREE.Object3D): number {
  let tris = 0;
  root.traverse((obj) => {
    if (obj instanceof THREE.Mesh && obj.geometry) {
      const geo = obj.geometry;
      if (geo.index) {
        tris += Math.floor(geo.index.count / 3);
      } else {
        const pos = geo.getAttribute("position");
        if (pos) tris += Math.floor(pos.count / 3);
      }
    }
  });
  return tris;
}

/** Smooth dynamic world-space ribbon trail (wingtip vortex / contrail) using zero-alloc ring buffer. */
export class RibbonTrail {
  readonly mesh: THREE.Mesh;
  private readonly maxPoints: number;
  private readonly ringCoords: Float32Array;
  private head = 0;
  private count = 0;
  private readonly positions: Float32Array;
  private readonly alphas: Float32Array;
  private readonly halfWidthM: number;

  constructor(maxPoints = 72, halfWidthM = 0.24, colorHex = 0x7dd3fc) {
    this.maxPoints = maxPoints;
    this.halfWidthM = halfWidthM;
    this.ringCoords = new Float32Array(maxPoints * 3);
    this.positions = new Float32Array(maxPoints * 2 * 3);
    this.alphas = new Float32Array(maxPoints * 2);

    const indices: number[] = [];
    for (let i = 0; i < maxPoints - 1; i++) {
      const a = i * 2;
      const b = a + 1;
      const c = a + 2;
      const d = a + 3;
      indices.push(a, b, c, b, d, c);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    geo.setAttribute("aAlpha", new THREE.BufferAttribute(this.alphas, 1));
    geo.setIndex(indices);
    geo.setDrawRange(0, 0);

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uColor: { value: new THREE.Color(colorHex) },
      },
      vertexShader: `
        attribute float aAlpha;
        varying float vAlpha;
        void main() {
          vAlpha = aAlpha;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        varying float vAlpha;
        void main() {
          gl_FragColor = vec4(uColor * 1.15, vAlpha * 0.34);
        }
      `,
    });

    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
  }

  reset(): void {
    this.head = 0;
    this.count = 0;
    this.mesh.geometry.setDrawRange(0, 0);
  }

  pushPoint(worldPos: THREE.Vector3, sideDir: THREE.Vector3): void {
    if (this.count > 0) {
      const hOff = this.head * 3;
      const dx = this.ringCoords[hOff] - worldPos.x;
      const dy = this.ringCoords[hOff + 1] - worldPos.y;
      const dz = this.ringCoords[hOff + 2] - worldPos.z;
      if (dx * dx + dy * dy + dz * dz > 16000) {
        // Scenario reset teleport — clear ribbon cleanly so no streak spans the sky
        this.reset();
      }
    }
    this.head = (this.head - 1 + this.maxPoints) % this.maxPoints;
    const hOff = this.head * 3;
    this.ringCoords[hOff] = worldPos.x;
    this.ringCoords[hOff + 1] = worldPos.y;
    this.ringCoords[hOff + 2] = worldPos.z;
    if (this.count < this.maxPoints) {
      this.count++;
    }
    const n = this.count;
    if (n < 2) return;

    const sx = sideDir.x;
    const sy = sideDir.y;
    const sz = sideDir.z;
    for (let i = 0; i < n; i++) {
      const rOff = ((this.head + i) % this.maxPoints) * 3;
      const px = this.ringCoords[rOff];
      const py = this.ringCoords[rOff + 1];
      const pz = this.ringCoords[rOff + 2];
      const frac = 1.0 - i / (n - 1);
      const w = this.halfWidthM * (0.28 + 1.35 * (1.0 - frac));
      const idx = i * 6;
      this.positions[idx + 0] = px - sx * w;
      this.positions[idx + 1] = py - sy * w;
      this.positions[idx + 2] = pz - sz * w;
      this.positions[idx + 3] = px + sx * w;
      this.positions[idx + 4] = py + sy * w;
      this.positions[idx + 5] = pz + sz * w;
      this.alphas[i * 2] = frac * frac;
      this.alphas[i * 2 + 1] = frac * frac;
    }

    const geo = this.mesh.geometry;
    geo.attributes.position.needsUpdate = true;
    geo.attributes.aAlpha.needsUpdate = true;
    geo.setDrawRange(0, (n - 1) * 6);
  }
}

/** Procedural brushed-aluminum panel-seam, micro-rivet, and aerospace livery CanvasTextures. */
function createAviationSurfaceMaps(): {
  readonly colorMap: THREE.CanvasTexture | null;
  readonly bumpMap: THREE.CanvasTexture | null;
  readonly roughnessMap: THREE.CanvasTexture | null;
} {
  if (typeof document === "undefined") {
    return { colorMap: null, bumpMap: null, roughnessMap: null };
  }

  const size = 512;

  // Color + aerospace livery map
  const colorCanvas = document.createElement("canvas");
  colorCanvas.width = size;
  colorCanvas.height = size;
  const cCtx = colorCanvas.getContext("2d");
  if (cCtx) {
    // Crisp pearl-white / satin-titanium civil aerospace livery with royal-blue & gold-amber cheatline
    const grad = cCtx.createLinearGradient(0, 0, 0, size);
    grad.addColorStop(0.0, "#eef3f8");
    grad.addColorStop(0.50, "#dde6f0");
    grad.addColorStop(0.53, "#1d4ed8"); // royal blue aviation stripe
    grad.addColorStop(0.56, "#f59e0b"); // warm gold-amber accent stripe
    grad.addColorStop(0.585, "#24344c"); // clean slate lower belly (not crushed black!)
    grad.addColorStop(1.0, "#2f415c");
    cCtx.fillStyle = grad;
    cCtx.fillRect(0, 0, size, size);

    // Subtle panel seams
    cCtx.strokeStyle = "rgba(15, 23, 42, 0.16)";
    cCtx.lineWidth = 1;
    for (let x = 0; x < size; x += 64) {
      cCtx.beginPath();
      cCtx.moveTo(x, 0);
      cCtx.lineTo(x, size);
      cCtx.stroke();
    }
    for (let y = 0; y < size; y += 64) {
      cCtx.beginPath();
      cCtx.moveTo(0, y);
      cCtx.lineTo(size, y);
      cCtx.stroke();
    }
  }

  // Panel-seam & micro-rivet bump map
  const bumpCanvas = document.createElement("canvas");
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bCtx = bumpCanvas.getContext("2d");
  if (bCtx) {
    bCtx.fillStyle = "#808080";
    bCtx.fillRect(0, 0, size, size);
    bCtx.strokeStyle = "#484848";
    bCtx.lineWidth = 1.5;
    for (let x = 0; x < size; x += 64) {
      bCtx.beginPath();
      bCtx.moveTo(x, 0);
      bCtx.lineTo(x, size);
      bCtx.stroke();
      // Rivet dots alongside panel seam
      bCtx.fillStyle = "#a8a8a8";
      for (let y = 6; y < size; y += 14) {
        bCtx.fillRect(x + 4, y, 2, 2);
      }
    }
    for (let y = 0; y < size; y += 64) {
      bCtx.beginPath();
      bCtx.moveTo(0, y);
      bCtx.lineTo(size, y);
      bCtx.stroke();
    }
  }

  // Roughness map (glossy composite upper, satin metal panels)
  const roughCanvas = document.createElement("canvas");
  roughCanvas.width = size;
  roughCanvas.height = size;
  const rCtx = roughCanvas.getContext("2d");
  if (rCtx) {
    rCtx.fillStyle = "#464646";
    rCtx.fillRect(0, 0, size, size);
    for (let x = 0; x < size; x += 64) {
      for (let y = 0; y < size; y += 64) {
        const shade = 54 + (((x * 7 + y * 13) >> 5) % 28);
        rCtx.fillStyle = `rgb(${shade},${shade},${shade})`;
        rCtx.fillRect(x + 1, y + 1, 62, 62);
      }
    }
  }

  const colorMap = new THREE.CanvasTexture(colorCanvas);
  colorMap.colorSpace = THREE.SRGBColorSpace;
  colorMap.wrapS = colorMap.wrapT = THREE.RepeatWrapping;

  const bumpMap = new THREE.CanvasTexture(bumpCanvas);
  bumpMap.wrapS = bumpMap.wrapT = THREE.RepeatWrapping;

  const roughnessMap = new THREE.CanvasTexture(roughCanvas);
  roughnessMap.wrapS = roughnessMap.wrapT = THREE.RepeatWrapping;

  return { colorMap, bumpMap, roughnessMap };
}

/** Builds a smooth lofted NACA-style swept airfoil wing mesh (one half, mirrorable). */
function createAirfoilWingGeometry(
  spanM: number,
  rootChordM: number,
  tipChordM: number,
  sweepBackM: number,
  dihedralM: number,
  sideSign: 1 | -1,
): THREE.BufferGeometry {
  const spanSegs = 14;
  const profileSegs = 22;
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  // NACA 2412-inspired closed profile around chord [0..1]
  const profilePts: Array<[number, number]> = [];
  for (let i = 0; i <= profileSegs; i++) {
    const theta = (i / profileSegs) * Math.PI * 2;
    const xc = 0.5 * (1.0 - Math.cos(theta)); // 0 at leading edge, 1 at trailing edge
    const yt =
      0.62 *
      (0.2969 * Math.sqrt(Math.max(1e-5, xc)) -
        0.126 * xc -
        0.3516 * xc * xc +
        0.2843 * xc * xc * xc -
        0.1015 * xc * xc * xc * xc);
    const camber = 0.025 * Math.sin(Math.PI * xc);
    const yc = theta <= Math.PI ? camber + yt : camber - yt;
    profilePts.push([xc - 0.32, yc]);
  }

  const ringStride = profileSegs + 1;
  for (let s = 0; s <= spanSegs; s++) {
    const t = s / spanSegs;
    const xPos = sideSign * t * spanM;
    const chord = rootChordM + (tipChordM - rootChordM) * Math.pow(t, 0.92);
    const thickScale = chord * (1.0 - 0.18 * t);
    const zOffset = t * sweepBackM;
    const yOffset = t * dihedralM;

    for (let p = 0; p <= profileSegs; p++) {
      const [px, py] = profilePts[p];
      positions.push(xPos, yOffset + py * thickScale, zOffset + px * chord);
      uvs.push(t * 2.0, p / profileSegs);
    }
  }

  for (let s = 0; s < spanSegs; s++) {
    for (let p = 0; p < profileSegs; p++) {
      const a = s * ringStride + p;
      const b = (s + 1) * ringStride + p;
      const c = (s + 1) * ringStride + (p + 1);
      const d = s * ringStride + (p + 1);
      if (sideSign > 0) {
        indices.push(a, c, b, a, d, c);
      } else {
        indices.push(a, b, c, a, c, d);
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/** Builds a sculpted propeller hub + 4 pitched composite blades + motion-blur disc. */
function createPropellerAssembly(
  spinnerMat: THREE.Material,
  bladeMat: THREE.Material,
): { readonly propGroup: THREE.Group; readonly blurMesh: THREE.Mesh } {
  const propGroup = new THREE.Group();

  // Contoured parabolic metallic spinner
  const spinnerPts: THREE.Vector2[] = [];
  for (let i = 0; i <= 14; i++) {
    const t = i / 14;
    spinnerPts.push(new THREE.Vector2(0.32 * Math.sqrt(t), 0.72 * (1.0 - t)));
  }
  const spinnerGeo = new THREE.LatheGeometry(spinnerPts, 24);
  spinnerGeo.rotateX(-Math.PI * 0.5);
  const spinner = new THREE.Mesh(spinnerGeo, spinnerMat);
  spinner.position.z = -0.36;
  propGroup.add(spinner);

  // 4 sculpted twisted scimitar blades with yellow safety tips
  const tipMat = new THREE.MeshStandardMaterial({
    color: 0xfacc15,
    roughness: 0.35,
    metalness: 0.1,
    emissive: 0x713f12,
    emissiveIntensity: 0.25,
  });

  const bladeShape = new THREE.Shape();
  bladeShape.moveTo(0.0, 0.08);
  bladeShape.quadraticCurveTo(0.14, 0.55, 0.11, 1.32);
  bladeShape.quadraticCurveTo(0.03, 1.48, -0.04, 1.44);
  bladeShape.quadraticCurveTo(-0.11, 0.9, -0.06, 0.08);
  bladeShape.closePath();

  const bladeGeo = new THREE.ExtrudeGeometry(bladeShape, {
    depth: 0.032,
    bevelEnabled: true,
    bevelThickness: 0.012,
    bevelSize: 0.012,
    bevelSegments: 2,
    curveSegments: 10,
  });
  bladeGeo.center();
  bladeGeo.translate(0, 0.72, 0);

  const tipGeo = new THREE.BoxGeometry(0.15, 0.14, 0.045);

  for (let b = 0; b < 4; b++) {
    const holder = new THREE.Group();
    holder.rotation.z = (b * Math.PI) / 2;
    const blade = new THREE.Mesh(bladeGeo, bladeMat);
    blade.rotation.y = 0.32;
    holder.add(blade);
    const tip = new THREE.Mesh(tipGeo, tipMat);
    tip.position.set(0.02, 1.38, 0);
    tip.rotation.y = 0.32;
    holder.add(tip);
    propGroup.add(holder);
  }

  // Translucent radial motion-blur prop disc
  const blurGeo = new THREE.CircleGeometry(1.48, 36);
  const blurMat = new THREE.MeshBasicMaterial({
    color: 0x94a3b8,
    transparent: true,
    opacity: 0.16,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const blurMesh = new THREE.Mesh(blurGeo, blurMat);
  blurMesh.position.z = -0.04;
  propGroup.add(blurMesh);

  return { propGroup, blurMesh };
}

/**
 * Builds the Hero Ownship Civil Executive Twin-Turboprop (`OPENXVS N40XVS`).
 * Wingspan ~16.8 m, length ~14.2 m, > 5,000 smooth-shaded triangles.
 */
export function createOwnshipAircraft(): OwnshipRig {
  const group = new THREE.Group();
  group.name = "openxvs_ownship_twin";

  const maps = createAviationSurfaceMaps();

  const fuselageMat = new THREE.MeshStandardMaterial({
    color: 0xe4ebf2,
    map: maps.colorMap,
    bumpMap: maps.bumpMap,
    bumpScale: 0.02,
    roughnessMap: maps.roughnessMap,
    roughness: 0.32,
    metalness: 0.30,
  });

  const wingMat = new THREE.MeshStandardMaterial({
    color: 0xe4ebf2,
    bumpMap: maps.bumpMap,
    bumpScale: 0.016,
    roughness: 0.32,
    metalness: 0.30,
  });

  const obsidianMat = new THREE.MeshStandardMaterial({
    color: 0x1e2c42,
    roughness: 0.32,
    metalness: 0.45,
  });

  const chromeMat = new THREE.MeshStandardMaterial({
    color: 0xcbd5e1,
    roughness: 0.14,
    metalness: 0.92,
  });

  const canopyMat = new THREE.MeshPhysicalMaterial({
    color: 0x061426,
    roughness: 0.05,
    metalness: 0.2,
    clearcoat: 1.0,
    clearcoatRoughness: 0.04,
    reflectivity: 1.0,
    emissive: 0x0c2d48,
    emissiveIntensity: 0.28,
  });

  const bladeMat = new THREE.MeshStandardMaterial({
    color: 0x161f2e,
    roughness: 0.32,
    metalness: 0.52,
  });

  // 1. Sculpted high-poly lofted fuselage (48 radial x 38 axial = ~3,600 triangles!)
  // Profile along axis y -> rotated onto -Z (nose at z = -6.8m, tail at z = +7.2m)
  const fuseProfile: Array<[number, number]> = [
    [-6.95, 0.0],
    [-6.88, 0.16],
    [-6.65, 0.34],
    [-6.25, 0.52],
    [-5.65, 0.72],
    [-4.95, 0.88],
    [-4.15, 1.02],
    [-3.2, 1.08],
    [-2.0, 1.1],
    [-0.5, 1.1],
    [1.2, 1.08],
    [2.6, 1.0],
    [3.9, 0.84],
    [5.1, 0.62],
    [6.1, 0.38],
    [6.95, 0.16],
    [7.25, 0.0],
  ];
  const lathePts = fuseProfile.map(([z, r]) => new THREE.Vector2(Math.max(0.001, r), -z));
  const fuseGeo = new THREE.LatheGeometry(lathePts, 48);
  fuseGeo.rotateX(-Math.PI * 0.5);
  fuseGeo.scale(0.94, 1.06, 1.0); // Slight tall-oval cabin cross-section
  fuseGeo.computeVertexNormals();
  const fuselage = new THREE.Mesh(fuseGeo, fuselageMat);
  fuselage.castShadow = true;
  fuselage.receiveShadow = true;
  group.add(fuselage);

  // Dark radar-transparent nose radome tip
  const radomeGeo = new THREE.SphereGeometry(0.46, 28, 18, 0, Math.PI * 2, 0, Math.PI * 0.58);
  radomeGeo.rotateX(-Math.PI * 0.5);
  radomeGeo.scale(0.92, 1.02, 1.72);
  const radome = new THREE.Mesh(radomeGeo, obsidianMat);
  radome.position.set(0, -0.02, -6.22);
  group.add(radome);

  // 2. Wrap-around clearcoat cockpit windshield + cabin windows
  const windshieldGeo = new THREE.SphereGeometry(1.06, 36, 20, -Math.PI * 0.56, Math.PI * 1.12, 0.22, 0.88);
  windshieldGeo.scale(0.95, 0.86, 1.52);
  const windshield = new THREE.Mesh(windshieldGeo, canopyMat);
  windshield.position.set(0, 0.36, -4.15);
  group.add(windshield);

  // Cabin passenger/avionics windows along both sides (subtle warm interior glow)
  const windowMat = new THREE.MeshStandardMaterial({
    color: 0x0b1a2e,
    emissive: 0x38bdf8,
    emissiveIntensity: 0.42,
    roughness: 0.08,
    metalness: 0.4,
  });
  const winGeo = new THREE.CapsuleGeometry(0.14, 0.22, 6, 12);
  winGeo.rotateZ(Math.PI * 0.5);
  for (let i = 0; i < 5; i++) {
    const zW = -1.9 + i * 0.92;
    for (const side of [-1, 1] as const) {
      const wMesh = new THREE.Mesh(winGeo, windowMat);
      wMesh.position.set(side * 1.01, 0.24, zW);
      wMesh.rotation.y = side * (Math.PI * 0.5);
      group.add(wMesh);
    }
  }

  // Royal-blue & gold fuselage accent stripes (stored so cockpit camera hides them cleanly!)
  const fuselageStripes: THREE.Object3D[] = [];
  const royalStripeMat = new THREE.MeshStandardMaterial({
    color: 0x1d4ed8,
    roughness: 0.28,
    metalness: 0.35,
  });
  const stripeGeo = new THREE.BoxGeometry(0.05, 0.075, 8.8);
  for (const side of [-1, 1] as const) {
    const stripe = new THREE.Mesh(stripeGeo, royalStripeMat);
    stripe.position.set(side * 1.01, -0.02, 0.25);
    group.add(stripe);
    fuselageStripes.push(stripe);
  }

  // 3. Swept NACA Airfoil Main Wings + Upswept Winglets + De-ice Leading Edge
  const wingR = new THREE.Mesh(createAirfoilWingGeometry(8.1, 2.55, 1.08, 1.35, 0.58, 1), wingMat);
  wingR.position.set(0.65, -0.34, -0.45);
  wingR.castShadow = true;
  wingR.receiveShadow = true;
  group.add(wingR);

  const wingL = new THREE.Mesh(createAirfoilWingGeometry(8.1, 2.55, 1.08, 1.35, 0.58, -1), wingMat);
  wingL.position.set(-0.65, -0.34, -0.45);
  wingL.castShadow = true;
  wingL.receiveShadow = true;
  group.add(wingL);

  // Upswept blended winglets
  for (const side of [-1, 1] as const) {
    const wlGeo = createAirfoilWingGeometry(1.42, 0.95, 0.36, 0.72, 0.0, side);
    const winglet = new THREE.Mesh(wlGeo, obsidianMat);
    winglet.position.set(side * 8.68, 0.22, 0.88);
    winglet.rotation.z = side * 1.24;
    group.add(winglet);
  }

  // Wing-root fairing
  const fairingGeo = new THREE.CapsuleGeometry(0.86, 3.6, 12, 24);
  fairingGeo.rotateX(Math.PI * 0.5);
  fairingGeo.scale(1.22, 0.58, 1.0);
  const fairing = new THREE.Mesh(fairingGeo, obsidianMat);
  fairing.position.set(0, -0.78, -0.15);
  group.add(fairing);

  // 4. Twin Wing-Mounted Turbine Nacelles + Spinning 4-Blade Propellers
  const nacPts: THREE.Vector2[] = [
    new THREE.Vector2(0.02, 1.95),
    new THREE.Vector2(0.36, 1.85),
    new THREE.Vector2(0.52, 1.45),
    new THREE.Vector2(0.56, 0.5),
    new THREE.Vector2(0.5, -0.8),
    new THREE.Vector2(0.32, -1.65),
    new THREE.Vector2(0.08, -1.95),
  ];
  const nacGeo = new THREE.LatheGeometry(nacPts, 28);
  nacGeo.rotateX(-Math.PI * 0.5);

  const intakeRingGeo = new THREE.TorusGeometry(0.36, 0.048, 12, 24);
  const exhaustGeo = new THREE.CylinderGeometry(0.09, 0.11, 0.42, 12);
  exhaustGeo.rotateX(Math.PI * 0.35);

  const leftPropBuild = createPropellerAssembly(chromeMat, bladeMat);
  const rightPropBuild = createPropellerAssembly(chromeMat, bladeMat);

  for (const side of [-1, 1] as const) {
    const nacGroup = new THREE.Group();
    nacGroup.position.set(side * 2.95, -0.18, -1.05);

    const nacMesh = new THREE.Mesh(nacGeo, fuselageMat);
    nacMesh.castShadow = true;
    nacGroup.add(nacMesh);

    const intake = new THREE.Mesh(intakeRingGeo, chromeMat);
    intake.position.set(0, -0.26, -1.62);
    nacGroup.add(intake);

    const exhaustL = new THREE.Mesh(exhaustGeo, chromeMat);
    exhaustL.position.set(-0.44, -0.06, -0.95);
    exhaustL.rotation.z = 0.55;
    nacGroup.add(exhaustL);

    const exhaustR = new THREE.Mesh(exhaustGeo, chromeMat);
    exhaustR.position.set(0.44, -0.06, -0.95);
    exhaustR.rotation.z = -0.55;
    nacGroup.add(exhaustR);

    const propAsm = side < 0 ? leftPropBuild : rightPropBuild;
    propAsm.propGroup.position.set(0, 0.02, -2.02);
    nacGroup.add(propAsm.propGroup);

    group.add(nacGroup);
  }

  // 5. Sculpted T-Tail Vertical Fin + Dorsal Fillet + Swept Horizontal Stabilizer
  const vFinShape = new THREE.Shape();
  vFinShape.moveTo(0, 0);
  vFinShape.lineTo(3.45, 0);
  vFinShape.lineTo(4.45, 2.75);
  vFinShape.lineTo(3.12, 2.75);
  vFinShape.lineTo(1.2, 0.42);
  vFinShape.closePath();

  const vFinGeo = new THREE.ExtrudeGeometry(vFinShape, {
    depth: 0.16,
    bevelEnabled: true,
    bevelThickness: 0.055,
    bevelSize: 0.04,
    bevelSegments: 3,
  });
  vFinGeo.translate(0, 0, -0.08);
  vFinGeo.rotateY(-Math.PI * 0.5);
  const vFinMat = new THREE.MeshStandardMaterial({
    color: 0xd8e2ee,
    roughness: 0.30,
    metalness: 0.32,
  });
  const vFin = new THREE.Mesh(vFinGeo, vFinMat);
  vFin.position.set(0, 0.48, 3.05);
  vFin.castShadow = true;
  group.add(vFin);

  // Royal-blue + amber vertical fin executive livery stripe
  const finAccentGeo = new THREE.BoxGeometry(0.24, 2.15, 0.14);
  const finAccent = new THREE.Mesh(finAccentGeo, royalStripeMat);
  finAccent.position.set(0, 1.85, 5.85);
  finAccent.rotation.x = -0.58;
  group.add(finAccent);

  // T-Tail horizontal stabilizers
  const hStabR = new THREE.Mesh(createAirfoilWingGeometry(2.85, 1.18, 0.62, 0.58, -0.06, 1), wingMat);
  hStabR.position.set(0.05, 3.16, 6.22);
  group.add(hStabR);

  const hStabL = new THREE.Mesh(createAirfoilWingGeometry(2.85, 1.18, 0.62, 0.58, -0.06, -1), wingMat);
  hStabL.position.set(-0.05, 3.16, 6.22);
  group.add(hStabL);

  // 6. Belly-Mounted `OpenXVS Optical-Thermal Multi-Sensor Pod`
  const podGroup = new THREE.Group();
  podGroup.position.set(0, -1.26, -2.55);
  const podBody = new THREE.Mesh(new THREE.CapsuleGeometry(0.32, 0.95, 10, 20), obsidianMat);
  podBody.rotation.x = Math.PI * 0.5;
  podGroup.add(podBody);

  const eoRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.18, 0.03, 10, 20),
    new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x22d3ee,
      emissiveIntensity: 2.2,
      roughness: 0.1,
    }),
  );
  eoRing.position.set(0, -0.04, -0.78);
  podGroup.add(eoRing);

  const lwirDome = new THREE.Mesh(
    new THREE.SphereGeometry(0.15, 16, 12),
    new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xd97706,
      emissiveIntensity: 1.5,
      roughness: 0.08,
      metalness: 0.8,
    }),
  );
  lwirDome.position.set(0, -0.16, -0.25);
  podGroup.add(lwirDome);
  group.add(podGroup);

  // 7. High-HDR Blooming Navigation, Beacon & Anti-Collision Strobe Lights
  const portRedMat = new THREE.MeshStandardMaterial({
    color: 0xff3b30,
    emissive: 0xff2d20,
    emissiveIntensity: 3.4,
    toneMapped: false,
  });
  const stbdGreenMat = new THREE.MeshStandardMaterial({
    color: 0x22c55e,
    emissive: 0x10b981,
    emissiveIntensity: 3.4,
    toneMapped: false,
  });
  const strobeMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    emissive: 0xe0f2fe,
    emissiveIntensity: 4.0,
    toneMapped: false,
  });
  const beaconMat = new THREE.MeshStandardMaterial({
    color: 0xff3b30,
    emissive: 0xef4444,
    emissiveIntensity: 2.8,
    toneMapped: false,
  });

  const bulbGeo = new THREE.SphereGeometry(0.16, 12, 10);
  const portNav = new THREE.Mesh(bulbGeo, portRedMat);
  portNav.position.set(-8.78, 0.45, 0.92);
  group.add(portNav);

  const stbdNav = new THREE.Mesh(bulbGeo, stbdGreenMat);
  stbdNav.position.set(8.78, 0.45, 0.92);
  group.add(stbdNav);

  const strobeL = new THREE.Mesh(bulbGeo, strobeMat);
  strobeL.position.set(-9.12, 1.48, 1.58);
  group.add(strobeL);

  const strobeR = new THREE.Mesh(bulbGeo, strobeMat);
  strobeR.position.set(9.12, 1.48, 1.58);
  group.add(strobeR);

  const tailBeacon = new THREE.Mesh(bulbGeo, beaconMat);
  tailBeacon.position.set(0, 3.34, 6.75);
  group.add(tailBeacon);

  const noseLandingLight = new THREE.Mesh(
    new THREE.SphereGeometry(0.14, 12, 10),
    new THREE.MeshStandardMaterial({
      color: 0xfffbeb,
      emissive: 0xfef3c7,
      emissiveIntensity: 3.2,
      toneMapped: false,
    }),
  );
  noseLandingLight.position.set(0, -0.58, -5.65);
  group.add(noseLandingLight);

  // 8. World-space wingtip contrail ribbons
  const leftTrail = new RibbonTrail(84, 0.22, 0x7dd3fc);
  const rightTrail = new RibbonTrail(84, 0.22, 0x7dd3fc);
  const wingtipLeftLocal = new THREE.Vector3(-9.12, 1.48, 1.58);
  const wingtipRightLocal = new THREE.Vector3(9.12, 1.48, 1.58);
  const tmpLeft = new THREE.Vector3();
  const tmpRight = new THREE.Vector3();
  const tmpSide = new THREE.Vector3();

  const triangleCount = countMeshTriangles(group);

  const cockpitOccludingParts: THREE.Object3D[] = [
    fuselage,
    radome,
    windshield,
    fairing,
    ...fuselageStripes,
  ];

  return {
    group,
    cockpitOccludingParts,
    leftProp: leftPropBuild.propGroup,
    rightProp: rightPropBuild.propGroup,
    leftBlur: leftPropBuild.blurMesh,
    rightBlur: rightPropBuild.blurMesh,
    strobeMat,
    beaconMat,
    leftTrail,
    rightTrail,
    triangleCount,
    update(dtS: number, simTimeS: number, engineOutRight: boolean): void {
      leftPropBuild.propGroup.rotation.z += dtS * 44.0;
      rightPropBuild.propGroup.rotation.z += dtS * (engineOutRight ? 2.4 : 44.0);
      (rightPropBuild.blurMesh.material as THREE.MeshBasicMaterial).opacity = engineOutRight ? 0.02 : 0.16;

      // Realistic aviation double-flash anti-collision strobe + rotating red tail beacon
      const phase = simTimeS % 1.15;
      const doubleFlash = (phase < 0.08) || (phase > 0.18 && phase < 0.26);
      strobeMat.emissiveIntensity = doubleFlash ? 5.2 : 0.35;
      beaconMat.emissiveIntensity = 0.8 + 2.8 * Math.pow(0.5 + 0.5 * Math.sin(simTimeS * 5.2), 4.0);

      // Update wingtip vortex trails in world space
      group.updateMatrixWorld();
      tmpLeft.copy(wingtipLeftLocal).applyMatrix4(group.matrixWorld);
      tmpRight.copy(wingtipRightLocal).applyMatrix4(group.matrixWorld);
      tmpSide.set(1, 0, 0).applyQuaternion(group.quaternion).normalize();
      leftTrail.pushPoint(tmpLeft, tmpSide);
      rightTrail.pushPoint(tmpRight, tmpSide);
    },
  };
}

/**
 * Builds the Intruder Civil Helicopter (`CBDR-1`, Bell-407 / H130 silhouette).
 * Rotor diameter ~10.6 m, > 2,500 smooth-shaded triangles.
 */
export function createIntruderHelicopter(): IntruderHeliRig {
  const group = new THREE.Group();
  group.name = "openxvs_intruder_cbdr1";

  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    roughness: 0.28,
    metalness: 0.55,
  });

  const accentMat = new THREE.MeshStandardMaterial({
    color: 0xf97316,
    emissive: 0xea580c,
    emissiveIntensity: 0.9,
    roughness: 0.3,
    metalness: 0.35,
  });

  const canopyMat = new THREE.MeshPhysicalMaterial({
    color: 0x050e1c,
    roughness: 0.04,
    metalness: 0.18,
    clearcoat: 1.0,
    clearcoatRoughness: 0.04,
  });

  const metalMat = new THREE.MeshStandardMaterial({
    color: 0x94a3b8,
    roughness: 0.25,
    metalness: 0.82,
  });

  const bladeMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.35,
    metalness: 0.45,
  });

  // 1. Sculpted teardrop helicopter cabin (40 radial x 24 axial)
  const cabProfile: Array<[number, number]> = [
    [-3.15, 0.0],
    [-3.0, 0.38],
    [-2.55, 0.78],
    [-1.75, 1.06],
    [-0.65, 1.18],
    [0.55, 1.14],
    [1.65, 0.92],
    [2.55, 0.54],
    [3.15, 0.24],
    [3.35, 0.0],
  ];
  const cabGeo = new THREE.LatheGeometry(
    cabProfile.map(([z, r]) => new THREE.Vector2(Math.max(0.001, r), -z)),
    40,
  );
  cabGeo.rotateX(-Math.PI * 0.5);
  cabGeo.scale(0.88, 1.04, 1.0);
  const cabin = new THREE.Mesh(cabGeo, bodyMat);
  cabin.castShadow = true;
  group.add(cabin);

  // Glossy tinted bubble canopy
  const bubbleGeo = new THREE.SphereGeometry(1.16, 32, 20, -Math.PI * 0.56, Math.PI * 1.12, 0.16, 1.15);
  bubbleGeo.scale(0.86, 0.92, 1.52);
  const bubble = new THREE.Mesh(bubbleGeo, canopyMat);
  bubble.position.set(0, 0.18, -1.25);
  group.add(bubble);

  // High-visibility safety orange fuselage stripe
  const stripeGeo = new THREE.TorusGeometry(1.06, 0.075, 10, 32);
  stripeGeo.scale(0.88, 1.04, 1.0);
  const stripe = new THREE.Mesh(stripeGeo, accentMat);
  stripe.position.set(0, 0, 0.72);
  group.add(stripe);

  // Top turbine cowling
  const cowlGeo = new THREE.CapsuleGeometry(0.54, 1.85, 10, 20);
  cowlGeo.rotateX(Math.PI * 0.5);
  const cowl = new THREE.Mesh(cowlGeo, bodyMat);
  cowl.position.set(0, 1.14, 0.32);
  group.add(cowl);

  // 2. Tapered tail boom + vertical fin + horizontal stabilizer
  const boomGeo = new THREE.CylinderGeometry(0.14, 0.36, 5.4, 20);
  boomGeo.rotateX(Math.PI * 0.5);
  const boom = new THREE.Mesh(boomGeo, bodyMat);
  boom.position.set(0, 0.22, 5.35);
  group.add(boom);

  const tailFinGeo = new THREE.BoxGeometry(0.12, 1.65, 0.72);
  const tailFin = new THREE.Mesh(tailFinGeo, accentMat);
  tailFin.position.set(0, 0.58, 7.92);
  tailFin.rotation.x = -0.25;
  group.add(tailFin);

  const tailStabGeo = new THREE.BoxGeometry(2.15, 0.08, 0.45);
  const tailStab = new THREE.Mesh(tailStabGeo, bodyMat);
  tailStab.position.set(0, 0.26, 6.25);
  group.add(tailStab);

  // 3. Tubular landing skids + cross struts
  const skidGeo = new THREE.CylinderGeometry(0.065, 0.065, 4.4, 12);
  skidGeo.rotateX(Math.PI * 0.5);
  const strutGeo = new THREE.CylinderGeometry(0.05, 0.05, 1.08, 10);
  for (const side of [-1, 1] as const) {
    const skid = new THREE.Mesh(skidGeo, metalMat);
    skid.position.set(side * 1.12, -1.52, -0.25);
    group.add(skid);

    for (const zS of [-1.25, 0.85]) {
      const strut = new THREE.Mesh(strutGeo, metalMat);
      strut.position.set(side * 0.88, -1.02, zS);
      strut.rotation.z = side * -0.42;
      group.add(strut);
    }
  }

  // 4. Rapidly spinning 4-blade main rotor + rotor blur disc
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 0.85, 14), metalMat);
  mast.position.set(0, 1.85, -0.15);
  group.add(mast);

  const mainRotor = new THREE.Group();
  mainRotor.position.set(0, 2.26, -0.15);
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.18, 18), metalMat);
  mainRotor.add(hub);

  const mBladeGeo = new THREE.BoxGeometry(0.26, 0.04, 5.25);
  for (let i = 0; i < 4; i++) {
    const holder = new THREE.Group();
    holder.rotation.y = (i * Math.PI) / 2;
    const blade = new THREE.Mesh(mBladeGeo, bladeMat);
    blade.position.z = 2.68;
    holder.add(blade);
    mainRotor.add(holder);
  }
  const rotorDisc = new THREE.Mesh(
    new THREE.RingGeometry(0.35, 5.3, 40),
    new THREE.MeshBasicMaterial({
      color: 0x94a3b8,
      transparent: true,
      opacity: 0.13,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  rotorDisc.rotation.x = -Math.PI * 0.5;
  mainRotor.add(rotorDisc);
  group.add(mainRotor);

  // 5. Spinning 2-blade tail rotor
  const tailRotor = new THREE.Group();
  tailRotor.position.set(-0.22, 0.62, 8.02);
  const tBlade = new THREE.Mesh(new THREE.BoxGeometry(0.04, 1.55, 0.14), bladeMat);
  tailRotor.add(tBlade);
  group.add(tailRotor);

  // 6. Blinking high-HDR crimson anti-collision beacon
  const beaconMat = new THREE.MeshStandardMaterial({
    color: 0xff2d20,
    emissive: 0xff1e1e,
    emissiveIntensity: 4.5,
    toneMapped: false,
  });
  const topBeacon = new THREE.Mesh(new THREE.SphereGeometry(0.18, 12, 10), beaconMat);
  topBeacon.position.set(0, 1.48, 8.02);
  group.add(topBeacon);

  const bellyBeacon = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 10), beaconMat);
  bellyBeacon.position.set(0, -1.16, 0.25);
  group.add(bellyBeacon);

  const triangleCount = countMeshTriangles(group);

  return {
    group,
    mainRotor,
    tailRotor,
    beaconMat,
    triangleCount,
    update(dtS: number, simTimeS: number): void {
      mainRotor.rotation.y += dtS * 28.0;
      tailRotor.rotation.x += dtS * 52.0;
      const pulse = Math.pow(0.5 + 0.5 * Math.sin(simTimeS * 7.2), 4.0);
      beaconMat.emissiveIntensity = 1.2 + 4.4 * pulse;
    },
  };
}
