/**
 * OpenXVS Cesium Viewport Controller.
 * Conforms strictly to docs/xvs/CONVENTIONS.md v1.0 and docs/xvs/WP01_SIM_RIG.md §2.4, §2.5.
 */

window.xvsWidget = null;
window.xvsProfile = null;
window.xvsImageryProvider = null;

function fnv1a(bytes) {
  let hash = 0x811c9dc5;
  const u32 = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const len = u32.length;
  for (let i = 0; i < len; i++) {
    const val = u32[i];
    hash ^= (val & 0xFF);
    hash = Math.imul(hash, 0x01000193);
    hash ^= ((val >>> 8) & 0xFF);
    hash = Math.imul(hash, 0x01000193);
    hash ^= ((val >>> 16) & 0xFF);
    hash = Math.imul(hash, 0x01000193);
    hash ^= (val >>> 24);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

window.initXVSViewer = function(profileName, port, maxLevel = 16, msaaOverride = null, profileConfig = null) {
  const container = document.getElementById("cesiumContainer");
  if (window.xvsWidget) {
    window.xvsWidget.destroy();
    window.xvsWidget = null;
  }

  window.xvsProfile = profileName;

  const isRgb = (profileName === "rgb");
  const isMarkers = (profileName === "markers");
  const isBinary = (profileName === "mask_earth_binary");
  const defaultMsaa = (profileConfig && profileConfig.msaa_samples !== undefined)
    ? profileConfig.msaa_samples
    : ((isRgb || profileName === "mask_earth") ? 4 : 1);
  const msaaSamples = (msaaOverride !== null && msaaOverride !== undefined) ? msaaOverride : defaultMsaa;

  const skyBoxEnabled = (profileConfig && profileConfig.sky_box !== undefined)
    ? Boolean(profileConfig.sky_box)
    : isRgb;
  const skyAtmosphereEnabled = (profileConfig && profileConfig.sky_atmosphere !== undefined)
    ? Boolean(profileConfig.sky_atmosphere)
    : isRgb;

  const widget = new Cesium.CesiumWidget(container, {
    baseLayer: false,
    terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    useDefaultRenderLoop: false,
    requestRenderMode: false,
    useBrowserRecommendedResolution: false,
    msaaSamples: msaaSamples,
    shadows: false,
    skyBox: skyBoxEnabled ? undefined : false,
    skyAtmosphere: skyAtmosphereEnabled ? undefined : false,
    contextOptions: {
      webgl: {
        preserveDrawingBuffer: true,
        alpha: false,
        antialias: false
      }
    }
  });

  const scene = widget.scene;
  scene.highDynamicRange = (profileConfig && profileConfig.hdr !== undefined) ? Boolean(profileConfig.hdr) : false;
  if (scene.postProcessStages && scene.postProcessStages.fxaa) {
    scene.postProcessStages.fxaa.enabled = (profileConfig && profileConfig.fxaa !== undefined) ? Boolean(profileConfig.fxaa) : false;
  }

  // C12: drawing-buffer and canvas client size guard
  const dw = scene.drawingBufferWidth;
  const dh = scene.drawingBufferHeight;
  const canvas = widget.canvas;
  const cw = canvas ? canvas.clientWidth : 0;
  const ch = canvas ? canvas.clientHeight : 0;
  if (dw !== 1024 || dh !== 1024 || cw !== 1024 || ch !== 1024) {
    widget.destroy();
    window.xvsWidget = null;
    throw new Error(
      `Canvas / drawing buffer size guard failed: drawingBuffer=(${dw}, ${dh}), clientSize=(${cw}, ${ch}), expected (1024, 1024)`
    );
  }

  // Disable screen space camera controller inputs
  const ssc = scene.screenSpaceCameraController;
  ssc.enableRotate = false;
  ssc.enableTranslate = false;
  ssc.enableZoom = false;
  ssc.enableTilt = false;
  ssc.enableLook = false;
  ssc.enableCollisionDetection = false;

  const globe = scene.globe;
  globe.tileCacheSize = 1000;
  globe.depthTestAgainstTerrain = (profileConfig && profileConfig.depth_test_against_terrain !== undefined)
    ? Boolean(profileConfig.depth_test_against_terrain)
    : false;
  globe.maximumScreenSpaceError = (profileConfig && profileConfig.maximum_screen_space_error !== undefined)
    ? profileConfig.maximum_screen_space_error
    : 2;

  // Clear any existing layers
  scene.imageryLayers.removeAll();

  const useImagery = (profileConfig && profileConfig.imagery !== undefined)
    ? (profileConfig.imagery === "usgs")
    : isRgb;

  if (useImagery) {
    const tileUrl = `http://127.0.0.1:${port}/tiles/usgs_imagery/{z}/{y}/{x}`;
    window.xvsImageryProvider = new Cesium.UrlTemplateImageryProvider({
      url: tileUrl,
      tilingScheme: new Cesium.WebMercatorTilingScheme(),
      maximumLevel: maxLevel,
      tileWidth: 256,
      tileHeight: 256,
      credit: "USDA, USGS The National Map: Orthoimagery"
    });
    scene.imageryLayers.addImageryProvider(window.xvsImageryProvider);
  }

  // Configure globe and scene appearance
  if (profileConfig) {
    globe.show = Boolean(profileConfig.globe_show);
    globe.showGroundAtmosphere = Boolean(profileConfig.ground_atmosphere);
    globe.enableLighting = Boolean(profileConfig.lighting);
    scene.fog.enabled = Boolean(profileConfig.fog);

    if (profileConfig.base_color) {
      const bc = profileConfig.base_color;
      globe.baseColor = new Cesium.Color(bc[0], bc[1], bc[2], bc[3]);
    }
    if (profileConfig.background_color === "sky") {
      scene.backgroundColor = undefined;
    } else if (Array.isArray(profileConfig.background_color)) {
      const bg = profileConfig.background_color;
      scene.backgroundColor = new Cesium.Color(bg[0], bg[1], bg[2], bg[3]);
    }
  } else {
    // Fallback defaults
    if (isRgb) {
      globe.baseColor = new Cesium.Color(0.0, 0.1, 0.2, 1.0);
      globe.showGroundAtmosphere = true;
      globe.enableLighting = true;
      scene.fog.enabled = true;
      globe.show = true;
    } else if (isMarkers) {
      globe.show = false;
      scene.backgroundColor = Cesium.Color.BLACK;
      globe.showGroundAtmosphere = false;
      globe.enableLighting = false;
      scene.fog.enabled = false;
    } else {
      globe.baseColor = Cesium.Color.WHITE;
      scene.backgroundColor = Cesium.Color.BLACK;
      globe.showGroundAtmosphere = false;
      globe.enableLighting = false;
      scene.fog.enabled = false;
      globe.show = true;
    }
  }

  window.xvsWidget = widget;

  // Resolve actual applied settings
  const actualBaseColor = globe.baseColor
    ? [globe.baseColor.red, globe.baseColor.green, globe.baseColor.blue, globe.baseColor.alpha]
    : [0.0, 0.0, 0.0, 1.0];
  const sceneBgColorRgba = scene.backgroundColor
    ? [scene.backgroundColor.red, scene.backgroundColor.green, scene.backgroundColor.blue, scene.backgroundColor.alpha]
    : [0.0, 0.0, 0.0, 1.0];
  const isSkyBg = Boolean(profileConfig && profileConfig.background_color === "sky");
  const actualBgColor = isSkyBg ? "sky" : sceneBgColorRgba;
  const actualSkyBox = Boolean(scene.skyBox);
  const actualSkyAtmosphere = Boolean(scene.skyAtmosphere);
  const actualSun = Boolean(scene.sun);
  const actualMoon = Boolean(scene.moon);
  const actualMsaa = scene.msaaSamples;

  return {
    success: true,
    profile: profileName,
    drawingBufferWidth: dw,
    drawingBufferHeight: dh,
    drawing_buffer: [dw, dh],
    msaa_samples: actualMsaa,
    msaaSamples: actualMsaa,
    sky_box: actualSkyBox,
    skyBox: actualSkyBox,
    sky_atmosphere: actualSkyAtmosphere,
    skyAtmosphere: actualSkyAtmosphere,
    sun: actualSun,
    moon: actualMoon,
    sun_moon: actualSun && actualMoon,
    globe_show: globe.show,
    base_color: actualBaseColor,
    background_color: actualBgColor,
    background_color_rgba: sceneBgColorRgba,
    background_label: isSkyBg ? "sky" : null,
    ground_atmosphere: globe.showGroundAtmosphere,
    lighting: globe.enableLighting,
    fog: scene.fog.enabled,
    fxaa: (scene.postProcessStages && scene.postProcessStages.fxaa) ? scene.postProcessStages.fxaa.enabled : false,
    hdr: scene.highDynamicRange,
    maximum_screen_space_error: globe.maximumScreenSpaceError,
    maximumScreenSpaceError: globe.maximumScreenSpaceError,
    depth_test_against_terrain: globe.depthTestAgainstTerrain,
    depthTestAgainstTerrain: globe.depthTestAgainstTerrain,
    logarithmic_depth_buffer: scene.logarithmicDepthBuffer,
    logarithmicDepthBuffer: scene.logarithmicDepthBuffer
  };
};

window.setXVSClock = function(isoUtcString) {
  if (!window.xvsWidget) return;
  const clock = window.xvsWidget.clock;
  clock.currentTime = Cesium.JulianDate.fromIso8601(isoUtcString);
  clock.shouldAnimate = false;
};

window.setXVSCamera = function(p, d, u, fov_rad, aspect_ratio) {
  const camera = window.xvsWidget.camera;
  camera.lookAtTransform(Cesium.Matrix4.IDENTITY);

  const P = new Cesium.Cartesian3(p[0], p[1], p[2]);
  const D = new Cesium.Cartesian3(d[0], d[1], d[2]);
  const U = new Cesium.Cartesian3(u[0], u[1], u[2]);

  camera.position = P;
  camera.direction = D;
  camera.up = U;
  camera.right = Cesium.Cartesian3.cross(D, U, new Cesium.Cartesian3());

  // A previous setXVSROIFrustum() call leaves a PerspectiveOffCenterFrustum on the camera.
  // Rig cameras always use a symmetric PerspectiveFrustum: re-install one if needed, otherwise
  // the .fov assignment below would silently do nothing (WP-02 G2 finding, Architecture Lead 2026-09-24).
  if (!(camera.frustum instanceof Cesium.PerspectiveFrustum)) {
    camera.frustum = new Cesium.PerspectiveFrustum({
      fov: fov_rad,
      aspectRatio: aspect_ratio,
      near: 0.5,
      far: 500000000.0
    });
  }
  camera.frustum.fov = fov_rad;
  camera.frustum.aspectRatio = aspect_ratio;
  camera.frustum.near = 0.5;
};

window.getXVSReadback = function() {
  const camera = window.xvsWidget.camera;
  const fov = (camera.frustum && camera.frustum.fov !== undefined) ? camera.frustum.fov : null;
  let aspect = null;
  if (camera.frustum) {
    if (camera.frustum.aspectRatio !== undefined) {
      aspect = camera.frustum.aspectRatio;
    } else if (camera.frustum.right !== undefined && camera.frustum.top !== undefined) {
      const w = camera.frustum.right - camera.frustum.left;
      const h = camera.frustum.top - camera.frustum.bottom;
      aspect = (h !== 0) ? (w / h) : 1.0;
    }
  }
  return {
    position: [camera.positionWC.x, camera.positionWC.y, camera.positionWC.z],
    direction: [camera.directionWC.x, camera.directionWC.y, camera.directionWC.z],
    up: [camera.upWC.x, camera.upWC.y, camera.upWC.z],
    right: [camera.rightWC.x, camera.rightWC.y, camera.rightWC.z],
    fov_rad: fov,
    aspect: aspect,
    near: camera.frustum.near,
    far: camera.frustum.far
  };
};

// --- Point Primitive / Marker Management (CONVENTIONS §6, WP01 §3 G2) ---
window.xvsPoints = null;

window.setXVSMarkers = function(markers) {
  const scene = window.xvsWidget.scene;
  if (!window.xvsPoints) {
    window.xvsPoints = scene.primitives.add(new Cesium.PointPrimitiveCollection());
  }
  window.xvsPoints.removeAll();

  for (let i = 0; i < markers.length; i++) {
    const m = markers[i];
    const rawP = m.p_ecef || m.position;
    const pos = new Cesium.Cartesian3(rawP[0], rawP[1], rawP[2]);
    const c = m.color || [1.0, 1.0, 1.0, 1.0];
    window.xvsPoints.add({
      position: pos,
      pixelSize: 6,
      color: new Cesium.Color(c[0], c[1], c[2], c[3] !== undefined ? c[3] : 1.0)
    });
  }
};

window.clearXVSMarkers = function() {
  if (window.xvsPoints) {
    window.xvsPoints.removeAll();
  }
};

window.getXVSMarkerProjections = function(markers) {
  const scene = window.xvsWidget.scene;
  const projections = [];
  for (let i = 0; i < markers.length; i++) {
    const m = markers[i];
    const rawP = m.p_ecef || m.position;
    const pos = new Cesium.Cartesian3(rawP[0], rawP[1], rawP[2]);
    const win = Cesium.SceneTransforms.worldToWindowCoordinates(scene, pos);
    if (win) {
      projections.push({
        id: m.id !== undefined ? m.id : i,
        u: win.x - 0.5,
        v: win.y - 0.5,
        x_cesium: win.x,
        y_cesium: win.y
      });
    } else {
      projections.push(null);
    }
  }
  return projections;
};

// --- In-Page Stability Calculation (WP01 §2.5, Decision D5, Note N2) ---
let stabilityState = {
  renderCount: 0,
  consecutiveMatches: 0,
  lastHash: null,
  prevPixels: null,
  lastPixels: null,
  lastMaxDiff: 0,
  lastNDiff: 0
};

window.resetXVSStability = function() {
  stabilityState.renderCount = 0;
  stabilityState.consecutiveMatches = 0;
  stabilityState.lastHash = null;
  stabilityState.prevPixels = null;
  stabilityState.lastPixels = null;
  stabilityState.lastMaxDiff = 0;
  stabilityState.lastNDiff = 0;
};

// D26a: requests the page's Cesium request scheduler still holds (active plus queued in its priority heap).
function xvsPendingRequests() {
  const rs = Cesium.RequestScheduler;
  return rs.statistics.numberOfActiveRequests + (rs.requestHeap ? rs.requestHeap.length : 0);
}

window.getXVSTileState = function() {
  const scene = window.xvsWidget.scene;
  return { tilesLoaded: scene.globe.tilesLoaded, globeShown: scene.globe.show, pendingRequests: xvsPendingRequests() };
};

window.stepXVSStability = function(backend, proxyIdle, requireNoPending) {
  window.xvsWidget.render();
  stabilityState.renderCount++;

  const scene = window.xvsWidget.scene;
  const tilesLoaded = (scene.globe.tilesLoaded || !scene.globe.show);
  const modelsReady = (!window.xvsModels || window.xvsModels.length === 0 || window.xvsModels.every(m => m.ready));
  // D26a (rgb only): no pending tile requests either; a frame is never compared or captured while tiles load
  const noPending = !requireNoPending || xvsPendingRequests() === 0;
  const ready = (stabilityState.renderCount >= 5) && tilesLoaded && proxyIdle && modelsReady && noPending;

  if (!ready) {
    stabilityState.consecutiveMatches = 0;
    stabilityState.lastHash = null;
    stabilityState.prevPixels = null;
    stabilityState.lastPixels = null;
    return {
      stable: false,
      renderCount: stabilityState.renderCount,
      consecutiveMatches: 0,
      maxDiff: 0,
      nDiff: 0,
      readyForComparison: false
    };
  }

  const context = scene.context;
  const pixels = context.readPixels({ x: 0, y: 0, width: 1024, height: 1024 });
  stabilityState.lastPixels = pixels;

  if (backend === "ref") {
    const h = fnv1a(pixels);
    if (stabilityState.lastHash !== null && h === stabilityState.lastHash) {
      stabilityState.consecutiveMatches++;
    } else {
      stabilityState.consecutiveMatches = 0;
    }
    stabilityState.lastHash = h;
    stabilityState.lastMaxDiff = 0;
    stabilityState.lastNDiff = 0;
    const isStable = (stabilityState.consecutiveMatches >= 3);
    return {
      stable: isStable,
      renderCount: stabilityState.renderCount,
      consecutiveMatches: stabilityState.consecutiveMatches,
      maxDiff: 0,
      nDiff: 0,
      readyForComparison: true
    };
  } else {
    // hw backend
    if (!stabilityState.prevPixels) {
      stabilityState.prevPixels = new Uint8Array(pixels);
      return {
        stable: false,
        renderCount: stabilityState.renderCount,
        consecutiveMatches: 0,
        maxDiff: 0,
        nDiff: 0,
        readyForComparison: true
      };
    }

    let maxDiff = 0;
    let nDiff = 0;
    const prev = stabilityState.prevPixels;
    const len = pixels.length;
    for (let i = 0; i < len; i += 4) {
      const dr = Math.abs(pixels[i] - prev[i]);
      const dg = Math.abs(pixels[i + 1] - prev[i + 1]);
      const db = Math.abs(pixels[i + 2] - prev[i + 2]);
      if (dr > 0 || dg > 0 || db > 0) {
        nDiff++;
        if (dr > maxDiff) maxDiff = dr;
        if (dg > maxDiff) maxDiff = dg;
        if (db > maxDiff) maxDiff = db;
      }
    }
    prev.set(pixels);
    stabilityState.lastMaxDiff = maxDiff;
    stabilityState.lastNDiff = nDiff;

    // D5: max_diff <= 1 AND n_diff <= 10486 (1% of 1024^2)
    if (maxDiff <= 1 && nDiff <= 10486) {
      stabilityState.consecutiveMatches++;
    } else {
      stabilityState.consecutiveMatches = 0;
    }
    const isStable = (stabilityState.consecutiveMatches >= 3);
    return {
      stable: isStable,
      renderCount: stabilityState.renderCount,
      consecutiveMatches: stabilityState.consecutiveMatches,
      maxDiff: maxDiff,
      nDiff: nDiff,
      readyForComparison: true
    };
  }
};

window.renderXVSFrame = function() {
  window.xvsWidget.render();
};

window.isXVSTilesLoaded = function() {
  const scene = window.xvsWidget.scene;
  return (scene.globe.tilesLoaded || !scene.globe.show);
};

window.readXVSPixelsHash = function() {
  const context = window.xvsWidget.scene.context;
  const pixels = context.readPixels({ x: 0, y: 0, width: 1024, height: 1024 });
  return fnv1a(pixels);
};

window.readXVSPixelsBase64 = function() {
  let pixels = stabilityState.lastPixels;
  if (!pixels) {
    const context = window.xvsWidget.scene.context;
    pixels = context.readPixels({ x: 0, y: 0, width: 1024, height: 1024 });
  } else {
    // Clear cached reference after consumption
    stabilityState.lastPixels = null;
  }

  let binary = "";
  const len = pixels.byteLength;
  const chunkSize = 8192;
  for (let i = 0; i < len; i += chunkSize) {
    binary += String.fromCharCode.apply(null, pixels.subarray(i, Math.min(i + chunkSize, len)));
  }
  return btoa(binary);
};

window.readXVSPixelsRGB = function() {
  let pixels = stabilityState.lastPixels;
  if (!pixels) {
    const context = window.xvsWidget.scene.context;
    pixels = context.readPixels({ x: 0, y: 0, width: 1024, height: 1024 });
  } else {
    stabilityState.lastPixels = null;
  }

  if (!window.xvsRgbBuffer) {
    window.xvsRgbBuffer = new Uint8Array(1024 * 1024 * 3);
  }
  const rgb = window.xvsRgbBuffer;
  for (let i = 0, j = 0; i < pixels.length; i += 4, j += 3) {
    rgb[j] = pixels[i];
    rgb[j + 1] = pixels[i + 1];
    rgb[j + 2] = pixels[i + 2];
  }

  let binary = "";
  const len = rgb.byteLength;
  const chunkSize = 8192;
  for (let i = 0; i < len; i += chunkSize) {
    binary += String.fromCharCode.apply(null, rgb.subarray(i, Math.min(i + chunkSize, len)));
  }
  return btoa(binary);
};

window.readXVSPixelsPNG = function() {
  let pixels = stabilityState.lastPixels;
  if (!pixels) {
    const context = window.xvsWidget.scene.context;
    pixels = context.readPixels({ x: 0, y: 0, width: 1024, height: 1024 });
  } else {
    stabilityState.lastPixels = null;
  }

  if (!window.xvsTransferCanvas) {
    window.xvsTransferCanvas = document.createElement("canvas");
    window.xvsTransferCanvas.width = 1024;
    window.xvsTransferCanvas.height = 1024;
    window.xvsTransferCtx = window.xvsTransferCanvas.getContext("2d", { willReadFrequently: false });
    window.xvsTransferImageData = window.xvsTransferCtx.createImageData(1024, 1024);
  }

  const ctx = window.xvsTransferCtx;
  const imgData = window.xvsTransferImageData;
  const dst = imgData.data;
  const rowBytes = 1024 * 4;

  for (let r = 0; r < 1024; r++) {
    const srcOff = (1023 - r) * rowBytes;
    const dstOff = r * rowBytes;
    dst.set(pixels.subarray(srcOff, srcOff + rowBytes), dstOff);
    for (let col = 0; col < 1024; col++) {
      dst[dstOff + col * 4 + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return window.xvsTransferCanvas.toDataURL("image/png").split(",")[1];
};

window.getWebGLDiagnostics = function() {
  const canvas = window.xvsWidget ? window.xvsWidget.canvas : document.createElement("canvas");
  const gl = window.xvsWidget ? window.xvsWidget.scene.context._gl : (canvas.getContext("webgl2") || canvas.getContext("webgl"));
  if (!gl) {
    return { success: false, error: "No WebGL context" };
  }
  const isWebGL2 = (typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext);
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  const vendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  const maxSamples = isWebGL2 ? gl.getParameter(gl.MAX_SAMPLES) : 1;

  return {
    success: true,
    isWebGL2: isWebGL2,
    unmaskedVendor: vendor,
    unmaskedRenderer: renderer,
    maxSamples: maxSamples
  };
};

// --- Depth Test Against Terrain Runtime Toggle (WP-02 §2.2, Item 5) ---
window.setXVSDepthTestAgainstTerrain = function(enabled) {
  if (window.xvsWidget && window.xvsWidget.scene && window.xvsWidget.scene.globe) {
    window.xvsWidget.scene.globe.depthTestAgainstTerrain = Boolean(enabled);
  }
};

// --- Traffic 3D Model Management (WP-02 §2.2, §2.3) ---
window.xvsModels = [];

window.addXVSModel = async function(params) {
  if (!window.xvsWidget) throw new Error("xvsWidget not initialized");
  const scene = window.xvsWidget.scene;
  const mat = params.modelMatrix ? Cesium.Matrix4.fromArray(params.modelMatrix) : Cesium.Matrix4.IDENTITY;

  const opts = {
    url: params.url,
    modelMatrix: mat,
    upAxis: Cesium.Axis.Z,
    forwardAxis: Cesium.Axis.X,
    scale: 1.0,
    minimumPixelSize: 0,
    shadows: Cesium.ShadowMode.DISABLED,
    allowPicking: false,
    backFaceCulling: false,
    // WP-02 v1.3 (D-B2b): no dynamic environment map. Cesium's default updates a model's environment map from the
    // scene over successive frames, so the lit shading depended on the page's render history (G3a determinism,
    // acceptance item 3: re-rendering a frame in a fresh process gave up to 6 DN on the model). Unlit masks are
    // unaffected.
    environmentMapOptions: { enabled: false }
  };

  if (params.customShader === "unlit_white") {
    opts.customShader = new Cesium.CustomShader({
      lightingModel: Cesium.LightingModel.UNLIT,
      fragmentShaderText: `
        void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
          material.diffuse = vec3(1.0, 1.0, 1.0);
          material.alpha = 1.0;
        }
      `
    });
  } else if (params.customShader === "unlit_black") {
    opts.customShader = new Cesium.CustomShader({
      lightingModel: Cesium.LightingModel.UNLIT,
      fragmentShaderText: `
        void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
          material.diffuse = vec3(0.0, 0.0, 0.0);
          material.alpha = 1.0;
        }
      `
    });
  } else if (params.customShader === "unlit_passthrough") {
    opts.customShader = new Cesium.CustomShader({
      lightingModel: Cesium.LightingModel.UNLIT,
      fragmentShaderText: `
        void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
        }
      `
    });
  }

  const model = await Cesium.Model.fromGltfAsync(opts);
  scene.primitives.add(model);
  window.xvsModels.push(model);
  return {
    index: window.xvsModels.length - 1,
    ready: model.ready
  };
};

window.clearXVSModels = function() {
  if (!window.xvsWidget) return;
  const scene = window.xvsWidget.scene;
  if (window.xvsModels) {
    for (let i = 0; i < window.xvsModels.length; i++) {
      const m = window.xvsModels[i];
      scene.primitives.remove(m);
      if (!m.isDestroyed()) {
        m.destroy();
      }
    }
  }
  window.xvsModels = [];
};

window.setXVSModelPose = function(index, modelMatrixArray) {
  if (window.xvsModels && window.xvsModels[index]) {
    window.xvsModels[index].modelMatrix = Cesium.Matrix4.fromArray(modelMatrixArray);
  }
};

window.setXVSModelShow = function(index, show) {
  if (window.xvsModels && window.xvsModels[index]) {
    window.xvsModels[index].show = Boolean(show);
  }
};

window.areXVSModelsReady = function() {
  if (!window.xvsModels || window.xvsModels.length === 0) return true;
  return window.xvsModels.every(m => m.ready);
};

// --- Perspective Off-Center Frustum for Ring Camera ROI (WP-02 §2.4) ---
window.setXVSROIFrustum = function(u0, v0, fx, fy, cx, cy, n, w, h) {
  if (!window.xvsWidget) return;
  n = (n !== undefined) ? n : 0.5;
  w = (w !== undefined) ? w : 1024;
  h = (h !== undefined) ? h : 1024;

  const camera = window.xvsWidget.camera;
  const cx_prime = cx - u0;
  const cy_prime = cy - v0;
  const left = n * (-0.5 - cx_prime) / fx;
  const right = n * (w - 0.5 - cx_prime) / fx;
  const top = n * (cy_prime + 0.5) / fy;
  const bottom = -n * (h - 0.5 - cy_prime) / fy;
  const frustum = new Cesium.PerspectiveOffCenterFrustum({
    left: left,
    right: right,
    top: top,
    bottom: bottom,
    near: n,
    far: 500000000.0
  });

  // Cesium's globe (QuadtreePrimitive) computes the screen-space error with
  // frustum.sseDenominator, which only PerspectiveFrustum defines (2 tan(fovy/2)).
  // PerspectiveOffCenterFrustum leaves it undefined, the SSE becomes NaN, no tile ever
  // "meets" it and the globe refines without bound (T-LIMB-ROI edge errors, renders that
  // never stabilise). For a pinhole of vertical focal length fy (px) and a drawing buffer
  // of h px, height / sseDenominator must equal fy, so sseDenominator = h / fy.
  const sseDen = h / fy;
  Object.defineProperty(frustum, 'sseDenominator', {
    value: sseDen, writable: false, configurable: true, enumerable: false
  });

  const origClone = frustum.clone.bind(frustum);
  frustum.clone = function(result) {
    result = origClone(result);
    if (!Object.prototype.hasOwnProperty.call(result, 'sseDenominator')) {
      Object.defineProperty(result, 'sseDenominator', {
        value: sseDen, writable: false, configurable: true, enumerable: false
      });
    }
    const initNear = n;
    const initLeft = left;
    const initRight = right;
    const initTop = top;
    const initBottom = bottom;

    delete result.near;
    Object.defineProperty(result, 'near', {
      get() {
        return this._scaledNear !== undefined ? this._scaledNear : initNear;
      },
      set(newNear) {
        this._scaledNear = newNear;
        const scale = (initNear > 0) ? (newNear / initNear) : 1.0;
        this.left = initLeft * scale;
        this.right = initRight * scale;
        this.top = initTop * scale;
        this.bottom = initBottom * scale;
      },
      configurable: true,
      enumerable: true
    });
    return result;
  };

  camera.frustum = frustum;
};
