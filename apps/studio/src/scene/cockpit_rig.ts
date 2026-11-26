/**
 * Fully Enclosed Garmin G1000 Twin-Turboprop Glass Cockpit (`OpenXVS Studio v5.0`).
 *
 * Mounted inside the ownship flight deck ONLY when Camera `4` (`Kokpit / Cockpit`) is active:
 *   - 100% sealed matte charcoal aerospace leather/carbon flight deck (full lower dashboard skirt,
 *     left/right cabin side walls, floor pan, deep center console & brow roof liner — ZERO ocean
 *     bleed-through and ZERO floating exterior bars!).
 *   - Recessed Garmin G1000 Twin Displays with dark metallic bezels & bottom softkey bars:
 *       1. Left PFD: Artificial horizon, magenta flight director chevron, airspeed/altitude/VSI tapes,
 *          compass tape, crosswind arrow, `MANUAL / AUTOPILOT / SHIELD OVERRIDE` annunciator.
 *       2. Right OpenXVS Safety MFD: Top-down radar with runway `28L` centerline, ownship chevron,
 *          intruder track, `XVS-ActiveRange` wedge, `SC3BF` `152.4 m` bubble, `ForcedLand` glide corridor,
 *          and `Kani 18/18` status bar.
 *   - Matte-charcoal rams-horn control yoke seated cleanly in front of lower-left pilot seat +
 *     center console with brushed-metal twin throttle & prop levers.
 *   - Subtle animated rain/fog droplets on windshield when weather demands.
 */

import * as THREE from "three";
import { ActiveRangeModeTs, type StudioSimSnapshot } from "../sim/studio_sim.ts";

export interface CockpitInteriorRig {
  readonly group: THREE.Group;
  readonly eyeLocalPos: THREE.Vector3;
  setVisible(visible: boolean): void;
  update(snap: StudioSimSnapshot, simTimeS: number): void;
}

function drawSoftkeyStrip(ctx: CanvasRenderingContext2D, labels: readonly string[]): void {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const stripH = 22;
  ctx.fillStyle = "#080e1a";
  ctx.fillRect(0, h - stripH, w, stripH);
  ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h - stripH);
  ctx.lineTo(w, h - stripH);
  ctx.stroke();

  const cellW = w / labels.length;
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "700 9px monospace";
  ctx.textAlign = "center";
  labels.forEach((lbl, idx) => {
    const x = idx * cellW;
    if (idx > 0) {
      ctx.beginPath();
      ctx.moveTo(x, h - stripH);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    ctx.fillText(lbl, x + cellW * 0.5, h - 7);
  });
}

function drawPfdCanvas(ctx: CanvasRenderingContext2D, snap: StudioSimSnapshot): void {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const cx = w * 0.5;
  const cy = h * 0.48;
  const botCut = 66;

  ctx.save();
  ctx.clearRect(0, 0, w, h);

  // 1. Artificial Horizon (Sky / Ground rotated by roll & shifted by pitch)
  ctx.save();
  ctx.beginPath();
  ctx.rect(58, 40, w - 122, h - 40 - botCut);
  ctx.clip();

  ctx.translate(cx, cy);
  ctx.rotate(-snap.ownship.rollRad);
  const pxPerRad = 220;
  const pitchShift = snap.ownship.pitchRad * pxPerRad;

  const skyGrad = ctx.createLinearGradient(0, -240 + pitchShift, 0, pitchShift);
  skyGrad.addColorStop(0, "#0b428a");
  skyGrad.addColorStop(1, "#2a7fd4");
  ctx.fillStyle = skyGrad;
  ctx.fillRect(-320, -340 + pitchShift, 640, 340);

  const gndGrad = ctx.createLinearGradient(0, pitchShift, 0, 240 + pitchShift);
  gndGrad.addColorStop(0, "#754c24");
  gndGrad.addColorStop(1, "#331f0e");
  ctx.fillStyle = gndGrad;
  ctx.fillRect(-320, pitchShift, 640, 340);

  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(-260, pitchShift);
  ctx.lineTo(260, pitchShift);
  ctx.stroke();

  // Pitch ladder (-20°..+20°)
  ctx.strokeStyle = "rgba(255, 255, 255, 0.88)";
  ctx.fillStyle = "#ffffff";
  ctx.font = "700 10px monospace";
  ctx.textAlign = "right";
  for (let deg = -20; deg <= 20; deg += 5) {
    if (deg === 0) continue;
    const y = pitchShift - ((deg * Math.PI) / 180) * pxPerRad;
    const hw = deg % 10 === 0 ? 42 : 20;
    ctx.lineWidth = deg % 10 === 0 ? 1.8 : 1.2;
    ctx.beginPath();
    ctx.moveTo(-hw, y);
    ctx.lineTo(hw, y);
    ctx.stroke();
    if (deg % 10 === 0) ctx.fillText(String(Math.abs(deg)), -hw - 5, y + 4);
  }
  ctx.restore();

  // G1000 Magenta Flight Director V-Bar Chevron
  ctx.strokeStyle = "#e879f9";
  ctx.lineWidth = 3.2;
  ctx.beginPath();
  ctx.moveTo(cx - 42, cy + 12);
  ctx.lineTo(cx, cy - 2);
  ctx.lineTo(cx + 42, cy + 12);
  ctx.stroke();

  // Yellow Fixed Aircraft Reference Wings
  ctx.strokeStyle = "#facc15";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx - 50, cy);
  ctx.lineTo(cx - 16, cy);
  ctx.lineTo(cx - 16, cy + 7);
  ctx.moveTo(cx + 50, cy);
  ctx.lineTo(cx + 16, cy);
  ctx.lineTo(cx + 16, cy + 7);
  ctx.stroke();
  ctx.fillStyle = "#facc15";
  ctx.fillRect(cx - 3, cy - 3, 6, 6);

  // 2. Left Airspeed Tape (kt)
  const iasKt = snap.ownship.tasMps * 1.94384;
  ctx.fillStyle = "rgba(6, 12, 24, 0.88)";
  ctx.fillRect(0, 40, 58, h - 40 - botCut);
  ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
  ctx.strokeRect(0, 40, 58, h - 40 - botCut);
  ctx.fillStyle = "#38bdf8";
  ctx.font = "800 15px monospace";
  ctx.textAlign = "center";
  ctx.fillText(`${Math.round(iasKt)}`, 29, cy + 5);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "700 9px monospace";
  ctx.fillText("IAS KT", 29, 54);

  // 3. Right Altitude Tape (ft) + VSI
  const altFt = Math.max(0, snap.ownship.pos.y * 3.28084);
  const vsiFpm = snap.ownship.vel.y * 196.8504;
  ctx.fillStyle = "rgba(6, 12, 24, 0.88)";
  ctx.fillRect(w - 64, 40, 64, h - 40 - botCut);
  ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
  ctx.strokeRect(w - 64, 40, 64, h - 40 - botCut);
  ctx.fillStyle = "#34d399";
  ctx.font = "800 14px monospace";
  ctx.fillText(`${Math.round(altFt)}`, w - 32, cy + 5);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "700 9px monospace";
  ctx.fillText("ALT FT", w - 32, 54);
  ctx.fillStyle = vsiFpm < -600 ? "#fbbf24" : "#cbd5e1";
  ctx.fillText(`${vsiFpm >= 0 ? "+" : ""}${Math.round(vsiFpm / 10) * 10}`, w - 32, cy + 28);

  // 4. Top G1000 Mode Annunciator Bar (+6 px top inner bezel margin)
  ctx.fillStyle = "#060c18";
  ctx.fillRect(0, 0, w, 40);
  let modeLabel = snap.controlMode === "manual" ? "MANUAL PILOT · FD ON" : "GFC 700 AUTOPILOT · AP";
  let modeCol = snap.controlMode === "manual" ? "#38bdf8" : "#34d399";
  if (snap.shieldOverrideActive) {
    modeLabel = `SHIELD OVERRIDE: ${snap.shieldReason}`;
    modeCol = "#10b981";
  } else if (snap.envelopeBreach) {
    modeLabel = `BREACH: ${snap.breachReason ?? "SHIELD OFF"}`;
    modeCol = "#f43f5e";
  }
  ctx.fillStyle = modeCol;
  ctx.font = "800 12.5px monospace";
  ctx.textAlign = "center";
  ctx.fillText(modeLabel, cx, 28);

  // 5. Heading & Crosswind Bar above G1000 Softkeys
  ctx.fillStyle = "#050b16";
  ctx.fillRect(0, h - botCut, w, botCut - 22);
  const hdgDeg = (((snap.ownship.headingRad * 180) / Math.PI) % 360 + 360) % 360;
  ctx.fillStyle = "#f8fafc";
  ctx.font = "800 13px monospace";
  ctx.textAlign = "center";
  ctx.fillText(`HDG ${String(Math.round(hdgDeg)).padStart(3, "0")}°`, cx, h - 38);

  ctx.fillStyle = "#38bdf8";
  ctx.font = "700 10px monospace";
  ctx.textAlign = "left";
  ctx.fillText(`XWIND ${Math.round(snap.weather.crosswindKt)} KT →`, 12, h - 38);
  ctx.textAlign = "right";
  ctx.fillStyle = snap.ownship.engineOutRight ? "#f43f5e" : "#34d399";
  ctx.fillText(snap.ownship.engineOutRight ? "R-ENG OUT" : "TWIN NOM", w - 12, h - 38);

  drawSoftkeyStrip(ctx, ["INSET", "PFD", "OBS", "CDI", "DME", "XPDR", "ALERTS"]);
  ctx.restore();
}

function drawSafetyMfdCanvas(ctx: CanvasRenderingContext2D, snap: StudioSimSnapshot): void {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const cx = w * 0.5;
  const cy = h * 0.58;
  const pxPerM = 0.082;

  ctx.save();
  ctx.fillStyle = "#050b16";
  ctx.fillRect(0, 0, w, h);

  // Range rings (500 m, 1000 m, 1500 m)
  ctx.strokeStyle = "rgba(56, 189, 248, 0.24)";
  ctx.lineWidth = 1.1;
  for (const rM of [500, 1000, 1500]) {
    ctx.beginPath();
    ctx.arc(cx, cy, rM * pxPerM, 0, Math.PI * 2);
    ctx.stroke();
  }

  const cosH = Math.cos(snap.ownship.headingRad);
  const sinH = Math.sin(snap.ownship.headingRad);
  const toRadar = (wx: number, wz: number): readonly [number, number] => {
    const dE = wx - snap.ownship.pos.x;
    const dN = -wz - -snap.ownship.pos.z;
    const ahead = dE * sinH + dN * cosH;
    const right = dE * cosH - dN * sinH;
    return [cx + right * pxPerM, cy - ahead * pxPerM];
  };

  // KSFO 28L runway centerline + threshold symbol
  const [rx0, ry0] = toRadar(0, 0);
  const [rx1, ry1] = toRadar(
    Math.sin((297.7 * Math.PI) / 180) * 3400,
    -Math.cos((297.7 * Math.PI) / 180) * 3400,
  );
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  ctx.moveTo(rx0, ry0);
  ctx.lineTo(rx1, ry1);
  ctx.stroke();
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "700 10px monospace";
  ctx.fillText("KSFO 28L", rx0 + 6, ry0 + 4);

  // ForcedLand trochoid corridor on MFD
  if (snap.forcedLandWaypoints.length > 2 && (snap.scenario === "forcedland" || snap.ownship.engineOutRight)) {
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    snap.forcedLandWaypoints.forEach((wp, idx) => {
      const [px, py] = toRadar(wp.pos.x, wp.pos.z);
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  // Intruder track + XVS-ActiveRange wedge + SC3BF 152.4 m safety bubble
  if (snap.intruderVisible) {
    const [ix, iy] = toRadar(snap.intruder.pos.x, snap.intruder.pos.z);
    const locked = snap.activeRange.mode === ActiveRangeModeTs.Locked || snap.activeRange.sigmaRM <= 95;
    const coneCol = locked ? "rgba(16, 185, 129, 0.30)" : "rgba(245, 158, 11, 0.30)";
    const ang = Math.atan2(iy - cy, ix - cx);
    const spread = locked ? 0.08 : 0.22;
    const rOuter = Math.min(210, (snap.activeRange.rHatM + snap.activeRange.sigmaRM * 0.4) * pxPerM);

    ctx.fillStyle = coneCol;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, rOuter, ang - spread, ang + spread);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = snap.sc3bfSafe ? "#22d3ee" : "#f43f5e";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.arc(ix, iy, 152.4 * pxPerM, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "#f97316";
    ctx.beginPath();
    ctx.moveTo(ix, iy - 5);
    ctx.lineTo(ix + 5, iy);
    ctx.lineTo(ix, iy + 5);
    ctx.lineTo(ix - 5, iy);
    ctx.closePath();
    ctx.fill();
  }

  // Ownship white chevron
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.moveTo(cx, cy - 8);
  ctx.lineTo(cx + 6, cy + 6);
  ctx.lineTo(cx, cy + 3);
  ctx.lineTo(cx - 6, cy + 6);
  ctx.closePath();
  ctx.fill();

  // Header & Status Bar above Softkeys (+6 px top / +8 px left & bottom bezel & throttle breathing room)
  ctx.fillStyle = "rgba(6, 12, 24, 0.94)";
  ctx.fillRect(0, 0, w, 40);
  ctx.fillStyle = "#38bdf8";
  ctx.font = "800 12px monospace";
  ctx.fillText("OPENXVS SAFETY MFD · 26/26 KANI", 20, 27);

  ctx.fillStyle = "rgba(6, 12, 24, 0.94)";
  ctx.fillRect(0, h - 62, w, 40);
  ctx.fillStyle = snap.shieldEnabled ? "#34d399" : "#f43f5e";
  ctx.font = "700 11px monospace";
  ctx.fillText(
    `SHIELD: ${snap.shieldEnabled ? "ARMED" : "OFF"} · R=${Math.round(snap.activeRange.rHatM)}m ±${Math.round(snap.activeRange.sigmaRM)}m`,
    20,
    h - 41,
  );

  drawSoftkeyStrip(ctx, ["MAP", "TRAFFIC", "TERRAIN", "XVS-AR", "SC3BF", "KANI", "CHKLIST"]);
  ctx.restore();
}

function createRainDropletsTexture(): THREE.CanvasTexture | null {
  if (typeof document === "undefined") return null;
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 128;
  const ctx = c.getContext("2d");
  if (!ctx) return null;
  ctx.clearRect(0, 0, 256, 128);
  for (let i = 0; i < 64; i++) {
    const x = ((i * 73) % 248) + 4;
    const y = ((i * 41) % 118) + 5;
    const r = 1.5 + (i % 3) * 0.9;
    ctx.fillStyle = "rgba(210, 232, 255, 0.42)";
    ctx.beginPath();
    ctx.ellipse(x, y, r, r * 2.2, 0.12, 0, Math.PI * 2);
    ctx.fill();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

export function createCockpitInteriorRig(): CockpitInteriorRig {
  const group = new THREE.Group();
  group.name = "openxvs_3d_cockpit";
  group.visible = false;

  // Pilot eye anchor inside enclosed Garmin G1000 flight deck
  const eyeLocalPos = new THREE.Vector3(0, 0.54, -2.68);

  const leatherMat = new THREE.MeshStandardMaterial({
    color: 0x141a24,
    roughness: 0.78,
    metalness: 0.18,
  });
  const panelMat = new THREE.MeshStandardMaterial({
    color: 0x18202e,
    roughness: 0.65,
    metalness: 0.34,
  });
  const bezelMat = new THREE.MeshStandardMaterial({
    color: 0x242e3f,
    roughness: 0.38,
    metalness: 0.62,
  });
  const trimMat = new THREE.MeshStandardMaterial({
    color: 0x334155,
    roughness: 0.42,
    metalness: 0.58,
  });
  const yokeMat = new THREE.MeshStandardMaterial({
    color: 0x1b222e,
    roughness: 0.52,
    metalness: 0.35,
  });
  const brushedMetalMat = new THREE.MeshStandardMaterial({
    color: 0xb8c4d4,
    roughness: 0.28,
    metalness: 0.85,
  });

  // Subtle overhead warm-neutral flight-deck dome illumination
  const deckLight = new THREE.PointLight(0xbfd8f5, 1.65, 3.8);
  deckLight.position.set(0, 0.95, -3.35);
  group.add(deckLight);

  // 1. Fully Enclosed Matte-Charcoal Cockpit Shell (ZERO ocean bleed or gaps anywhere!)
  // Sculpted anti-glare glareshield brow overhang
  const glareshield = new THREE.Mesh(new THREE.BoxGeometry(2.55, 0.055, 0.58), leatherMat);
  glareshield.position.set(0, 0.445, -4.02);
  group.add(glareshield);

  const browLip = new THREE.Mesh(new THREE.BoxGeometry(2.50, 0.020, 0.05), trimMat);
  browLip.position.set(0, 0.470, -3.73);
  group.add(browLip);

  // Full sealed instrument bulkhead extending all the way down to floor pan
  const instrBulkhead = new THREE.Mesh(new THREE.BoxGeometry(2.75, 1.38, 0.16), panelMat);
  instrBulkhead.position.set(0, -0.22, -4.22);
  group.add(instrBulkhead);

  // Lower knee bolster & sealed floor pan (prevents any sky/ocean bleed beneath dashboard)
  const floorPan = new THREE.Mesh(new THREE.BoxGeometry(2.75, 0.10, 2.40), leatherMat);
  floorPan.position.set(0, -0.82, -3.25);
  group.add(floorPan);

  // Left & Right enclosed padded cockpit side walls (seals left/right edges completely!)
  for (const side of [-1, 1] as const) {
    const sideWall = new THREE.Mesh(new THREE.BoxGeometry(0.14, 1.38, 2.20), leatherMat);
    sideWall.position.set(side * 1.18, -0.18, -3.32);
    group.add(sideWall);

    const sillTrim = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.045, 1.80), trimMat);
    sillTrim.position.set(side * 1.12, 0.42, -3.48);
    group.add(sillTrim);
  }

  // Overhead roof liner & panoramic windshield framing (unobstructed forward boresight at x=0!)
  const roofLiner = new THREE.Mesh(new THREE.BoxGeometry(2.50, 0.14, 1.65), leatherMat);
  roofLiner.position.set(0, 1.24, -3.35);
  group.add(roofLiner);

  // Ultra-slim offset canopy quarter-mullions (leaves forward target centerline 100% clear)
  const mullionGeo = new THREE.BoxGeometry(0.015, 0.90, 0.026);
  for (const side of [-1, 1] as const) {
    const mullion = new THREE.Mesh(mullionGeo, leatherMat);
    mullion.position.set(side * 0.64, 0.83, -4.26);
    mullion.rotation.set(0.42, side * -0.06, side * 0.04);
    group.add(mullion);
  }

  const sidePillarGeo = new THREE.BoxGeometry(0.028, 0.92, 0.042);
  for (const side of [-1, 1] as const) {
    const sidePost = new THREE.Mesh(sidePillarGeo, leatherMat);
    sidePost.position.set(side * 1.06, 0.80, -3.98);
    sidePost.rotation.set(0.38, side * -0.18, side * 0.14);
    group.add(sidePost);
  }

  // Subtle windshield rain/fog droplets sheet
  const rainTex = createRainDropletsTexture();
  const rainMat = new THREE.MeshBasicMaterial({
    map: rainTex,
    transparent: true,
    opacity: 0.0,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const rainGlass = new THREE.Mesh(new THREE.PlaneGeometry(2.12, 0.78), rainMat);
  rainGlass.position.set(0, 0.78, -4.22);
  rainGlass.rotation.x = 0.42;
  group.add(rainGlass);

  // 2. Recessed Garmin G1000 Twin Displays (Left PFD + Right OpenXVS Safety MFD)
  let pfdCanvas: HTMLCanvasElement | null = null;
  let mfdCanvas: HTMLCanvasElement | null = null;
  let pfdTex: THREE.CanvasTexture | null = null;
  let mfdTex: THREE.CanvasTexture | null = null;

  if (typeof document !== "undefined") {
    pfdCanvas = document.createElement("canvas");
    pfdCanvas.width = 440;
    pfdCanvas.height = 310;
    pfdTex = new THREE.CanvasTexture(pfdCanvas);
    pfdTex.colorSpace = THREE.SRGBColorSpace;

    mfdCanvas = document.createElement("canvas");
    mfdCanvas.width = 440;
    mfdCanvas.height = 310;
    mfdTex = new THREE.CanvasTexture(mfdCanvas);
    mfdTex.colorSpace = THREE.SRGBColorSpace;
  }

  const screenGeo = new THREE.PlaneGeometry(0.55, 0.37);
  const outerBezelGeo = new THREE.BoxGeometry(0.61, 0.43, 0.045);
  const innerRecessGeo = new THREE.BoxGeometry(0.57, 0.39, 0.02);

  for (const [xPos, tex] of [
    [-0.37, pfdTex],
    [0.37, mfdTex],
  ] as const) {
    const bezel = new THREE.Mesh(outerBezelGeo, bezelMat);
    bezel.position.set(xPos, 0.20, -4.12);
    bezel.rotation.x = -0.08;
    group.add(bezel);

    const recess = new THREE.Mesh(innerRecessGeo, leatherMat);
    recess.position.set(xPos, 0.20, -4.102);
    recess.rotation.x = -0.08;
    group.add(recess);

    const screen = new THREE.Mesh(
      screenGeo,
      new THREE.MeshBasicMaterial({ map: tex, color: 0xffffff }),
    );
    screen.position.set(xPos, 0.20, -4.088);
    screen.rotation.x = -0.08;
    group.add(screen);
  }

  // 3. Properly Seated Matte-Charcoal Rams-Horn Control Yoke (ZERO bright white shaft!)
  const yokePivot = new THREE.Group();
  yokePivot.position.set(-0.37, -0.14, -3.56);

  const yokeBoot = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.052, 0.10, 14), leatherMat);
  yokeBoot.rotation.x = Math.PI * 0.5;
  yokeBoot.position.z = -0.52;
  yokePivot.add(yokeBoot);

  const yokeCol = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.024, 0.56, 14), yokeMat);
  yokeCol.rotation.x = Math.PI * 0.5;
  yokeCol.position.z = -0.28;
  yokePivot.add(yokeCol);

  const yokeWheel = new THREE.Group();
  const hub = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.068, 0.042), bezelMat);
  yokeWheel.add(hub);
  const crossBar = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.032, 0.032), yokeMat);
  yokeWheel.add(crossBar);
  for (const side of [-1, 1] as const) {
    const grip = new THREE.Mesh(new THREE.CapsuleGeometry(0.022, 0.135, 6, 10), yokeMat);
    grip.position.set(side * 0.175, 0.065, 0);
    grip.rotation.z = side * -0.14;
    yokeWheel.add(grip);
  }
  yokePivot.add(yokeWheel);
  group.add(yokePivot);

  // 4. Recessed Center Console Pedestal + Brushed-Metal Twin Throttle & Prop Levers
  // Seated cleanly below both G1000 display sightlines so softkeys & annunciators stay 100% unobstructed.
  const consoleBase = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.56, 0.85), leatherMat);
  consoleBase.position.set(0.0, -0.46, -3.72);
  group.add(consoleBase);

  const consoleTop = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.032, 0.72), bezelMat);
  consoleTop.position.set(0.0, -0.17, -3.75);
  group.add(consoleTop);

  const makeThrottleLever = (xOff: number, heightM: number, knobHex: number): THREE.Group => {
    const piv = new THREE.Group();
    piv.position.set(xOff, -0.16, -3.74);
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, heightM, 10), brushedMetalMat);
    stem.position.y = heightM * 0.5;
    piv.add(stem);
    const knob = new THREE.Mesh(
      new THREE.CylinderGeometry(0.018, 0.018, 0.038, 12),
      new THREE.MeshStandardMaterial({ color: knobHex, roughness: 0.45, metalness: 0.35 }),
    );
    knob.rotation.z = Math.PI * 0.5;
    knob.position.y = heightM + 0.005;
    piv.add(knob);
    group.add(piv);
    return piv;
  };
  const leftThrottle = makeThrottleLever(-0.052, 0.105, 0x111827);
  const rightThrottle = makeThrottleLever(-0.016, 0.105, 0x111827);
  // Blue-knobbed prop RPM levers beside throttles
  makeThrottleLever(0.022, 0.090, 0x1d4ed8);
  makeThrottleLever(0.056, 0.090, 0x1d4ed8);

  let frameCounter = 0;

  return {
    group,
    eyeLocalPos,
    setVisible(visible: boolean): void {
      group.visible = visible;
    },
    update(snap: StudioSimSnapshot, simTimeS: number): void {
      if (!group.visible) return;

      yokeWheel.rotation.z = -snap.yokeRoll * 0.68;
      yokePivot.position.z = -3.56 - snap.yokePitch * 0.055;

      const leftAngle = 0.42 - snap.ownship.throttle * 0.85;
      const rightAngle = snap.ownship.engineOutRight ? 0.54 : leftAngle;
      leftThrottle.rotation.x = leftAngle;
      rightThrottle.rotation.x = rightAngle;

      const wetLevel = Math.max(
        snap.weather.rainIntensity,
        snap.fogDensity > 0.0003 ? Math.min(0.65, (snap.fogDensity - 0.0003) * 1100) : 0.0,
      );
      rainMat.opacity = wetLevel * 0.42;
      if (rainTex && wetLevel > 0.02) {
        rainTex.offset.y = -(simTimeS * 0.35) % 1.0;
      }

      frameCounter++;
      if (frameCounter >= 1) {
        if (pfdCanvas && pfdTex) {
          const ctx = pfdCanvas.getContext("2d");
          if (ctx) {
            drawPfdCanvas(ctx, snap);
            pfdTex.needsUpdate = true;
          }
        }
        if (mfdCanvas && mfdTex) {
          const ctx = mfdCanvas.getContext("2d");
          if (ctx) {
            drawSafetyMfdCanvas(ctx, snap);
            mfdTex.needsUpdate = true;
          }
        }
      }
    },
  };
}
