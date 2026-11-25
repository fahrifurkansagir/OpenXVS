/**
 * OpenXVS Studio v5.0 Core 3D PBR + Physical Atmosphere + 3D Cockpit Engine (`Three.js`).
 *
 * - Default Mode: `photoreal` (`Gerçek Dünya / Photorealistic`, Golden Hour Daylight 16.5h)
 *   with 1-click switch to `obsidian` (`Taktik Opsidyen`).
 * - Physical Preetham/Rayleigh+Mie sky dome + 3D Volumetric Cumulus Cloud Deck (`atmosphere_world.ts`).
 * - Throttled live `PMREMGenerator` IBL updates on sun-angle or mode changes (>= 60 FPS!).
 * - Restrained daylight bloom (`threshold: 0.92, strength: 0.24` in Photoreal vs `0.78 / 0.55` in Obsidian).
 * - Interactive 3D Cockpit Interior (`cockpit_rig.ts`) mounted inside ownship ONLY for Camera `4` (`Kokpit`).
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import {
  createIntruderHelicopter,
  createOwnshipAircraft,
  type IntruderHeliRig,
  type OwnshipRig,
} from "./aircraft_factory.ts";
import {
  createAtmosphereRig,
  type AtmosphereLightingState,
  type AtmosphereRig,
  type VisualWorldMode,
} from "./atmosphere_world.ts";
import { createCockpitInteriorRig, type CockpitInteriorRig } from "./cockpit_rig.ts";
import { createKsfoWorld, type KsfoWorldRig } from "./ksfo_world.ts";
import { createSafetyManifoldsRig, type SafetyManifoldsRig } from "./manifolds.ts";
import type { KsfoTerrainPack } from "../sim/ksfo_terrain_pack.ts";
import type { StudioSimSnapshot } from "../sim/studio_sim.ts";
import type { StudioCameraPreset } from "../ui/i18n.ts";

export interface StudioEngineStats {
  readonly ownshipTriangles: number;
  readonly intruderTriangles: number;
  readonly cameraPreset: StudioCameraPreset;
  readonly visualMode: VisualWorldMode;
  readonly sensorDomeVisible: boolean;
  readonly drawCalls: number;
}

/** Generates a dynamic PMREM environment map matched to the live solar atmosphere state. */
function buildAtmospherePmrem(
  pmrem: THREE.PMREMGenerator,
  lighting: AtmosphereLightingState,
  mode: VisualWorldMode,
): THREE.Texture {
  const envScene = new THREE.Scene();
  const skyGeo = new THREE.SphereGeometry(50, 20, 14);
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    uniforms: {
      uSunDir: { value: lighting.sunDir.clone() },
      uSunCol: { value: lighting.sunColor.clone() },
      uDaylight: { value: lighting.daylightFactor },
      uObsidian: { value: mode === "obsidian" ? 1.0 : 0.0 },
    },
    vertexShader: `
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 uSunDir;
      uniform vec3 uSunCol;
      uniform float uDaylight;
      uniform float uObsidian;
      varying vec3 vDir;
      void main() {
        float y = vDir.y;
        vec3 topCol = mix(vec3(0.05, 0.11, 0.22), vec3(0.18, 0.42, 0.78), uDaylight * (1.0 - uObsidian));
        vec3 horCol = mix(vec3(0.16, 0.34, 0.56), vec3(0.68, 0.78, 0.88), uDaylight * (1.0 - uObsidian));
        vec3 gndCol = mix(vec3(0.03, 0.06, 0.11), vec3(0.18, 0.24, 0.28), uDaylight * (1.0 - uObsidian));
        vec3 col = y > 0.0
          ? mix(horCol, topCol, pow(y, 0.55))
          : mix(horCol * 0.5, gndCol, pow(-y, 0.5));
        float sunSpot = pow(max(dot(normalize(vDir), normalize(uSunDir)), 0.0), 12.0);
        col += uSunCol * sunSpot * (0.45 + 0.55 * uDaylight);
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
  envScene.add(new THREE.Mesh(skyGeo, skyMat));
  const rt = pmrem.fromScene(envScene, 0.02);
  skyGeo.dispose();
  skyMat.dispose();
  return rt.texture;
}

export class StudioEngine {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly composer: EffectComposer;
  readonly bloomPass: UnrealBloomPass;
  readonly atmosphere: AtmosphereRig;
  readonly world: KsfoWorldRig;
  readonly ownship: OwnshipRig;
  readonly cockpit: CockpitInteriorRig;
  readonly intruder: IntruderHeliRig;
  readonly manifolds: SafetyManifoldsRig;

  private readonly pmrem: THREE.PMREMGenerator;
  private readonly hemiLight: THREE.HemisphereLight;
  private readonly sunLight: THREE.DirectionalLight;
  private readonly rimLight: THREE.DirectionalLight;
  private readonly sceneFog: THREE.FogExp2;
  private envTex: THREE.Texture | null = null;
  private lastIblSunElev = -999;
  private lastIblMode: VisualWorldMode | null = null;

  private visualMode: VisualWorldMode = "photoreal";
  private cameraPreset: StudioCameraPreset = "chase";
  private userOrbitOffset = new THREE.Vector3();
  private userDragging = false;
  private lastOwnPos = new THREE.Vector3();
  private initializedCam = false;

  // Scratch vectors
  private readonly desiredCamPos = new THREE.Vector3();
  private readonly desiredTarget = new THREE.Vector3();
  private readonly fwdVec = new THREE.Vector3();
  private readonly rightVec = new THREE.Vector3();
  private readonly upVec = new THREE.Vector3(0, 1, 0);
  private readonly cockpitWorldPos = new THREE.Vector3();
  private readonly cockpitLookWorld = new THREE.Vector3();

  constructor(canvas: HTMLCanvasElement) {
    const width = Math.max(320, canvas.clientWidth || window.innerWidth);
    const height = Math.max(240, canvas.clientHeight || window.innerHeight);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: "high-performance",
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.pmrem = new THREE.PMREMGenerator(this.renderer);
    this.pmrem.compileEquirectangularShader();

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xc9e0f2);
    this.sceneFog = new THREE.FogExp2(0xc9e0f2, 0.000048);
    this.scene.fog = this.sceneFog;
    this.scene.environmentIntensity = 1.1;

    this.camera = new THREE.PerspectiveCamera(48, width / height, 0.12, 48000);

    // Dynamic Lighting Rig driven every frame by AtmosphereRig
    this.hemiLight = new THREE.HemisphereLight(0x93c5fd, 0x334155, 1.1);
    this.scene.add(this.hemiLight);

    this.sunLight = new THREE.DirectionalLight(0xfff4e0, 2.5);
    this.sunLight.position.set(-1400, 620, -950);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.set(1024, 1024);
    this.sunLight.shadow.camera.near = 10;
    this.sunLight.shadow.camera.far = 3500;
    this.sunLight.shadow.camera.left = -90;
    this.sunLight.shadow.camera.right = 90;
    this.sunLight.shadow.camera.top = 90;
    this.sunLight.shadow.camera.bottom = -90;
    this.scene.add(this.sunLight);
    this.scene.add(this.sunLight.target);

    this.rimLight = new THREE.DirectionalLight(0x38bdf8, 0.75);
    this.rimLight.position.set(900, 480, 1100);
    this.scene.add(this.rimLight);

    // Atmosphere (Physical Sky + 3D Cumulus Cloud Deck), KSFO World, Aircraft, Cockpit & Manifolds
    this.atmosphere = createAtmosphereRig();
    this.scene.add(this.atmosphere.group);

    this.world = createKsfoWorld(null);
    this.world.setVisualMode(this.visualMode);
    this.scene.add(this.world.group);

    this.ownship = createOwnshipAircraft();
    this.scene.add(this.ownship.group);
    this.scene.add(this.ownship.leftTrail.mesh);
    this.scene.add(this.ownship.rightTrail.mesh);

    this.cockpit = createCockpitInteriorRig();
    this.ownship.group.add(this.cockpit.group);

    this.intruder = createIntruderHelicopter();
    this.scene.add(this.intruder.group);

    this.manifolds = createSafetyManifoldsRig();
    this.scene.add(this.manifolds.group);

    // Post-Processing: RenderPass -> Restrained Daylight / Crisp Obsidian UnrealBloomPass -> OutputPass
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(width, height), 0.24, 0.38, 0.92);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.maxDistance = 12000;
    this.controls.minDistance = 8;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.addEventListener("start", () => {
      this.userDragging = true;
    });
    this.controls.addEventListener("end", () => {
      this.userDragging = false;
      this.userOrbitOffset.subVectors(this.camera.position, this.controls.target);
    });

    window.addEventListener("resize", () => this.resize());
  }

  resize(): void {
    const canvas = this.renderer.domElement;
    const width = Math.max(320, canvas.clientWidth || window.innerWidth);
    const height = Math.max(240, canvas.clientHeight || window.innerHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.composer.setSize(width, height);
    this.bloomPass.setSize(width, height);
  }

  setTerrainPack(pack: KsfoTerrainPack): void {
    this.world.setTerrainPack(pack);
  }

  setOrthoTextures(core2k: THREE.Texture | null, inner10k: THREE.Texture | null): void {
    this.world.setOrthoTextures(core2k, inner10k);
  }

  setVisualMode(mode: VisualWorldMode): void {
    this.visualMode = mode;
    this.world.setVisualMode(mode);
    if (mode === "photoreal") {
      this.bloomPass.threshold = 0.92;
      this.bloomPass.strength = 0.24;
      this.bloomPass.radius = 0.38;
      this.renderer.toneMappingExposure = 1.08;
    } else {
      this.bloomPass.threshold = 0.78;
      this.bloomPass.strength = 0.55;
      this.bloomPass.radius = 0.42;
      this.renderer.toneMappingExposure = 1.15;
    }
  }

  getVisualMode(): VisualWorldMode {
    return this.visualMode;
  }

  setCameraPreset(preset: StudioCameraPreset, snapImmediate = false): void {
    this.cameraPreset = preset;
    this.userOrbitOffset.set(0, 0, 0);
    this.cockpit.setVisible(preset === "cockpit");
    if (snapImmediate) {
      this.initializedCam = false;
    }
  }

  getCameraPreset(): StudioCameraPreset {
    return this.cameraPreset;
  }

  resetTrails(): void {
    this.ownship.leftTrail.reset();
    this.ownship.rightTrail.reset();
  }

  getStats(): StudioEngineStats {
    return {
      ownshipTriangles: this.ownship.triangleCount,
      intruderTriangles: this.intruder.triangleCount,
      cameraPreset: this.cameraPreset,
      visualMode: this.visualMode,
      sensorDomeVisible: this.manifolds.isSensorDomeVisible(),
      drawCalls: this.renderer.info.render.calls,
    };
  }

  renderFrame(snap: StudioSimSnapshot, dtS: number): void {
    // 1. Update Ownship & Intruder 3D Pose (Local -Z forward -> rotation.y = -headingRad)
    this.ownship.group.position.set(snap.ownship.pos.x, snap.ownship.pos.y, snap.ownship.pos.z);
    this.ownship.group.rotation.set(
      snap.ownship.pitchRad,
      -snap.ownship.headingRad,
      -snap.ownship.rollRad,
      "YXZ",
    );
    this.ownship.update(dtS, snap.simTimeS, snap.ownship.engineOutRight);
    const isCockpit = this.cameraPreset === "cockpit";
    for (const part of this.ownship.cockpitOccludingParts) {
      part.visible = !isCockpit;
    }
    this.cockpit.setVisible(isCockpit);
    this.cockpit.update(snap, snap.simTimeS);

    this.intruder.group.visible = snap.intruderVisible;
    if (snap.intruderVisible) {
      this.intruder.group.position.set(snap.intruder.pos.x, snap.intruder.pos.y, snap.intruder.pos.z);
      this.intruder.group.rotation.set(
        snap.intruder.pitchRad,
        -snap.intruder.headingRad,
        -snap.intruder.rollRad,
        "YXZ",
      );
      this.intruder.update(dtS, snap.simTimeS);
    }

    // 2. Update Physical Atmosphere & Lighting + Throttled PMREM IBL
    const lighting = this.atmosphere.update(
      snap.simTimeS,
      dtS,
      this.camera.position,
      snap.weather,
      this.visualMode,
    );

    this.hemiLight.color.copy(lighting.hemiSkyColor);
    this.hemiLight.groundColor.copy(lighting.hemiGroundColor);
    this.hemiLight.intensity = lighting.hemiIntensity;

    this.sceneFog.color.copy(lighting.fogColor);
    this.sceneFog.density = snap.fogDensity;
    this.scene.background = lighting.fogColor;

    this.sunLight.color.copy(lighting.sunColor);
    this.sunLight.intensity = lighting.sunIntensity;
    this.sunLight.position
      .copy(this.ownship.group.position)
      .addScaledVector(lighting.sunDir, 680);
    this.sunLight.target.position.copy(this.ownship.group.position);

    this.rimLight.intensity = this.visualMode === "obsidian" ? 1.15 : 0.42;

    if (
      this.lastIblMode !== this.visualMode ||
      Math.abs(lighting.sunElevationDeg - this.lastIblSunElev) > 3.2
    ) {
      this.lastIblMode = this.visualMode;
      this.lastIblSunElev = lighting.sunElevationDeg;
      const nextEnv = buildAtmospherePmrem(this.pmrem, lighting, this.visualMode);
      this.scene.environment = nextEnv;
      this.envTex?.dispose();
      this.envTex = nextEnv;
    }

    // 3. Update Luminous 3D Safety Manifolds
    this.manifolds.update(snap, snap.simTimeS);

    // 4. Update Cinematic Camera Rig
    const ownPos = this.ownship.group.position;
    this.fwdVec.set(Math.sin(snap.ownship.headingRad), 0, -Math.cos(snap.ownship.headingRad)).normalize();
    this.rightVec.crossVectors(this.fwdVec, this.upVec).normalize();

    if (this.cameraPreset === "chase") {
      this.desiredCamPos
        .copy(ownPos)
        .addScaledVector(this.fwdVec, -34.0)
        .addScaledVector(this.rightVec, 15.5)
        .addScaledVector(this.upVec, 11.5);
      this.desiredTarget
        .copy(ownPos)
        .addScaledVector(this.fwdVec, 68.0)
        .addScaledVector(this.upVec, -3.5);
    } else if (this.cameraPreset === "wingman") {
      this.desiredCamPos
        .copy(ownPos)
        .addScaledVector(this.fwdVec, -13.5)
        .addScaledVector(this.rightVec, 24.5)
        .addScaledVector(this.upVec, 4.8);
      this.desiredTarget
        .copy(ownPos)
        .addScaledVector(this.fwdVec, 6.5)
        .addScaledVector(this.upVec, -0.5);
    } else if (this.cameraPreset === "overview") {
      this.desiredCamPos.set(1750, 960, 1580);
      this.desiredTarget.set(-680, 45, -340);
    } else {
      // Exact pilot eye inside the interactive 3D Cockpit looking cleanly over PFD/MFD + glareshield!
      this.ownship.group.updateMatrixWorld();
      this.cockpitWorldPos.copy(this.cockpit.eyeLocalPos).applyMatrix4(this.ownship.group.matrixWorld);
      this.cockpitLookWorld.set(0, -4.4, -38.0).applyMatrix4(this.ownship.group.matrixWorld);
      this.desiredCamPos.copy(this.cockpitWorldPos);
      this.desiredTarget.copy(this.cockpitLookWorld);
    }

    const teleported = this.initializedCam && this.lastOwnPos.distanceToSquared(ownPos) > 25000;
    if (!this.initializedCam || teleported) {
      this.camera.position.copy(this.desiredCamPos);
      this.controls.target.copy(this.desiredTarget);
      this.initializedCam = true;
    } else if (!this.userDragging) {
      if (this.cameraPreset === "cockpit") {
        this.camera.position.copy(this.desiredCamPos);
        this.controls.target.copy(this.desiredTarget);
      } else if (this.userOrbitOffset.lengthSq() > 1.0 && this.cameraPreset !== "overview") {
        this.controls.target.lerp(ownPos, Math.min(1.0, dtS * 5.0));
        this.camera.position.copy(this.controls.target).add(this.userOrbitOffset);
      } else {
        const alpha = Math.min(1.0, dtS * 4.2);
        this.camera.position.lerp(this.desiredCamPos, alpha);
        this.controls.target.lerp(this.desiredTarget, alpha);
      }
    }
    this.lastOwnPos.copy(ownPos);

    if (this.cameraPreset !== "cockpit") {
      this.controls.update();
    } else {
      // Roll cockpit camera smoothly with aircraft bank for true flight-deck immersion
      this.camera.up.set(0, 1, 0).applyQuaternion(this.ownship.group.quaternion);
      this.camera.lookAt(this.desiredTarget);
    }
    if (this.cameraPreset !== "cockpit") {
      this.camera.up.set(0, 1, 0);
    }

    // 5. Update KSFO World (water waves, terrain lighting, ALSF-2 running strobes, ATC radar + beacon)
    this.world.update(snap.simTimeS, lighting, snap.fogDensity);

    // 6. Render full PBR + restrained Bloom + ACES Filmic OutputPass
    this.composer.render();
  }
}
