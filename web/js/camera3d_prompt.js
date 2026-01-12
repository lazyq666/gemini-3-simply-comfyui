import { app } from "/scripts/app.js";

const EXTENSION_NAME = "Gemini3.Camera3DPrompt";
const NODE_NAME = "Gemini3Camera3DPrompt";
const DEFAULT_NODE_WIDTH = 520;
const DEFAULT_NODE_HEIGHT = 640;
const RESERVED_WIDGET_SPACE = 40;
const MIN_CAMERA_HEIGHT = 260;

const AZIMUTH_STEPS = [0, 45, 90, 135, 180, 225, 270, 315];
const ELEVATION_STEPS = [-30, 0, 30, 60];
const DISTANCE_STEPS = [0.6, 1.0, 1.8];

const AZIMUTH_NAMES = {
  0: "front view",
  45: "front-right quarter view",
  90: "right side view",
  135: "back-right quarter view",
  180: "back view",
  225: "back-left quarter view",
  270: "left side view",
  315: "front-left quarter view",
};

const ELEVATION_NAMES = {
  "-30": "low-angle shot",
  0: "eye-level shot",
  30: "elevated shot",
  60: "high-angle shot",
};

const DISTANCE_NAMES = {
  "0.6": "close-up",
  1: "medium shot",
  "1.8": "wide shot",
};

const THREE_URL = "https://unpkg.com/three@0.152.2/build/three.min.js";

function ensureStyles() {
  const styleId = "gemini3-camera3d-style";
  if (document.getElementById(styleId)) {
    return;
  }
  const style = document.createElement("style");
  style.id = styleId;
  style.textContent = `
    .gemini3-camera3d-root {
      position: relative;
      width: 100%;
      height: 100%;
      overflow: hidden;
    }
    .gemini3-camera3d-wrapper {
      position: relative;
      width: 100%;
      height: 100%;
      background: #1a1a1a;
      border-radius: 10px;
      overflow: hidden;
    }
    .gemini3-camera3d-overlay {
      position: absolute;
      bottom: 10px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(0, 0, 0, 0.8);
      padding: 6px 12px;
      border-radius: 8px;
      font-family: monospace;
      font-size: 12px;
      color: #00ff88;
      white-space: nowrap;
      z-index: 10;
      pointer-events: none;
    }
  `;
  document.head.appendChild(style);
}

function loadScriptOnce(url) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${url}"]`);
    if (existing) {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = url;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load ${url}`));
    document.head.appendChild(script);
  });
}

function chainCallback(object, property, callback) {
  if (!object) {
    return;
  }
  if (property in object) {
    const callbackOrig = object[property];
    object[property] = function () {
      const result = callbackOrig.apply(this, arguments);
      callback.apply(this, arguments);
      return result;
    };
  } else {
    object[property] = callback;
  }
}

function hideWidgetForGood(node, widget, suffix = "") {
  if (!widget) {
    return;
  }
  widget.hidden = true;
  widget.origType = widget.type;
  widget.origComputeSize = widget.computeSize;
  widget.origSerializeValue = widget.serializeValue;
  widget.computeSize = () => [0, -4];
  widget.type = `converted-widget${suffix}`;
  if (widget.linkedWidgets) {
    for (const linked of widget.linkedWidgets) {
      hideWidgetForGood(node, linked, `:${widget.name}`);
    }
  }
}

function hideCameraWidgets(node) {
  hideWidgetForGood(node, getWidget(node, "azimuth"));
  hideWidgetForGood(node, getWidget(node, "elevation"));
  hideWidgetForGood(node, getWidget(node, "distance"));
}

function snapToNearest(value, steps) {
  return steps.reduce(
    (prev, curr) =>
      Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev,
    steps[0]
  );
}

function buildPrompt(azimuth, elevation, distance) {
  const azSnap = snapToNearest(azimuth, AZIMUTH_STEPS);
  const elSnap = snapToNearest(elevation, ELEVATION_STEPS);
  const distSnap = snapToNearest(distance, DISTANCE_STEPS);
  const distKey = distSnap === 1 ? 1 : distSnap.toFixed(1);
  return `${AZIMUTH_NAMES[azSnap]} ${ELEVATION_NAMES[String(elSnap)]} ${DISTANCE_NAMES[distKey]}`;
}

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name) || null;
}

function readWidgetValue(node, name, fallback) {
  const widget = getWidget(node, name);
  if (!widget) {
    return fallback;
  }
  const value = Number(widget.value);
  return Number.isFinite(value) ? value : fallback;
}

function writeWidgetValue(node, name, value) {
  const widget = getWidget(node, name);
  if (!widget || widget.value === value) {
    return;
  }
  widget.value = value;
  if (node.graph) {
    node.graph._version++;
    node.setDirtyCanvas(true, true);
  }
}

async function createCameraWidget(node, element) {
  ensureStyles();
  if (!window.THREE) {
    try {
      await loadScriptOnce(THREE_URL);
    } catch (error) {
      console.error(error);
      return null;
    }
  }
  if (!window.THREE) {
    console.error("THREE not available; camera widget disabled.");
    return null;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "gemini3-camera3d-wrapper";
  const overlay = document.createElement("div");
  overlay.className = "gemini3-camera3d-overlay";
  wrapper.appendChild(overlay);
  element.classList.add("gemini3-camera3d-root");
  element.appendChild(wrapper);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
  camera.position.set(4.5, 3, 4.5);
  camera.lookAt(0, 0.75, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(wrapper.clientWidth || 1, wrapper.clientHeight || 1);
  wrapper.insertBefore(renderer.domElement, overlay);

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
  dirLight.position.set(5, 10, 5);
  scene.add(dirLight);
  scene.add(new THREE.GridHelper(8, 16, 0x333333, 0x222222));

  const CENTER = new THREE.Vector3(0, 0.75, 0);
  const BASE_DISTANCE = 1.6;
  const AZIMUTH_RADIUS = 2.4;
  const ELEVATION_RADIUS = 1.8;

  let azimuthAngle = readWidgetValue(node, "azimuth", 0);
  let elevationAngle = readWidgetValue(node, "elevation", 0);
  let distanceFactor = readWidgetValue(node, "distance", 1.0);

  function createPlaceholderTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#3a3a4a";
    ctx.fillRect(0, 0, 256, 256);
    ctx.fillStyle = "#ffcc99";
    ctx.beginPath();
    ctx.arc(128, 128, 80, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#333";
    ctx.beginPath();
    ctx.arc(100, 110, 10, 0, Math.PI * 2);
    ctx.arc(156, 110, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(128, 130, 35, 0.2, Math.PI - 0.2);
    ctx.stroke();
    return new THREE.CanvasTexture(canvas);
  }

  const planeMaterial = new THREE.MeshBasicMaterial({
    map: createPlaceholderTexture(),
    side: THREE.DoubleSide,
  });
  let targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
  targetPlane.position.copy(CENTER);
  scene.add(targetPlane);

  function updateTextureFromBase64(base64) {
    if (!base64) {
      planeMaterial.map = createPlaceholderTexture();
      planeMaterial.needsUpdate = true;
      scene.remove(targetPlane);
      targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
      targetPlane.position.copy(CENTER);
      scene.add(targetPlane);
      return;
    }
    const url = base64.startsWith("data:")
      ? base64
      : `data:image/jpeg;base64,${base64}`;
    const loader = new THREE.TextureLoader();
    loader.crossOrigin = "anonymous";
    loader.load(
      url,
      (texture) => {
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        planeMaterial.map = texture;
        planeMaterial.needsUpdate = true;
        const img = texture.image;
        if (img && img.width && img.height) {
          const aspect = img.width / img.height;
          const maxSize = 1.5;
          let planeWidth;
          let planeHeight;
          if (aspect > 1) {
            planeWidth = maxSize;
            planeHeight = maxSize / aspect;
          } else {
            planeHeight = maxSize;
            planeWidth = maxSize * aspect;
          }
          scene.remove(targetPlane);
          targetPlane = new THREE.Mesh(
            new THREE.PlaneGeometry(planeWidth, planeHeight),
            planeMaterial
          );
          targetPlane.position.copy(CENTER);
          scene.add(targetPlane);
        }
      },
      undefined,
      (error) => {
        console.error("Failed to load texture", error);
      }
    );
  }

  const cameraGroup = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0x6699cc,
    metalness: 0.5,
    roughness: 0.3,
  });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.22, 0.38), bodyMat);
  cameraGroup.add(body);
  const lens = new THREE.Mesh(
    new THREE.CylinderGeometry(0.09, 0.11, 0.18, 16),
    new THREE.MeshStandardMaterial({
      color: 0x6699cc,
      metalness: 0.5,
      roughness: 0.3,
    })
  );
  lens.rotation.x = Math.PI / 2;
  lens.position.z = 0.26;
  cameraGroup.add(lens);
  scene.add(cameraGroup);

  const azimuthRing = new THREE.Mesh(
    new THREE.TorusGeometry(AZIMUTH_RADIUS, 0.04, 16, 64),
    new THREE.MeshStandardMaterial({
      color: 0x00ff88,
      emissive: 0x00ff88,
      emissiveIntensity: 0.3,
    })
  );
  azimuthRing.rotation.x = Math.PI / 2;
  azimuthRing.position.y = 0.05;
  scene.add(azimuthRing);

  const azimuthHandle = new THREE.Mesh(
    new THREE.SphereGeometry(0.18, 16, 16),
    new THREE.MeshStandardMaterial({
      color: 0x00ff88,
      emissive: 0x00ff88,
      emissiveIntensity: 0.5,
    })
  );
  azimuthHandle.userData.type = "azimuth";
  scene.add(azimuthHandle);

  const arcPoints = [];
  for (let i = 0; i <= 32; i += 1) {
    const angle = THREE.MathUtils.degToRad(-30 + (90 * i) / 32);
    arcPoints.push(
      new THREE.Vector3(
        -0.8,
        ELEVATION_RADIUS * Math.sin(angle) + CENTER.y,
        ELEVATION_RADIUS * Math.cos(angle)
      )
    );
  }
  const arcCurve = new THREE.CatmullRomCurve3(arcPoints);
  const elevationArc = new THREE.Mesh(
    new THREE.TubeGeometry(arcCurve, 32, 0.04, 8, false),
    new THREE.MeshStandardMaterial({
      color: 0xff69b4,
      emissive: 0xff69b4,
      emissiveIntensity: 0.3,
    })
  );
  scene.add(elevationArc);

  const elevationHandle = new THREE.Mesh(
    new THREE.SphereGeometry(0.18, 16, 16),
    new THREE.MeshStandardMaterial({
      color: 0xff69b4,
      emissive: 0xff69b4,
      emissiveIntensity: 0.5,
    })
  );
  elevationHandle.userData.type = "elevation";
  scene.add(elevationHandle);

  const distanceLineGeo = new THREE.BufferGeometry();
  const distanceLine = new THREE.Line(
    distanceLineGeo,
    new THREE.LineBasicMaterial({ color: 0xffa500 })
  );
  scene.add(distanceLine);

  const distanceHandle = new THREE.Mesh(
    new THREE.SphereGeometry(0.18, 16, 16),
    new THREE.MeshStandardMaterial({
      color: 0xffa500,
      emissive: 0xffa500,
      emissiveIntensity: 0.5,
    })
  );
  distanceHandle.userData.type = "distance";
  scene.add(distanceHandle);

  function updatePositions() {
    const distance = BASE_DISTANCE * distanceFactor;
    const azRad = THREE.MathUtils.degToRad(azimuthAngle);
    const elRad = THREE.MathUtils.degToRad(elevationAngle);

    const camX = distance * Math.sin(azRad) * Math.cos(elRad);
    const camY = distance * Math.sin(elRad) + CENTER.y;
    const camZ = distance * Math.cos(azRad) * Math.cos(elRad);

    cameraGroup.position.set(camX, camY, camZ);
    cameraGroup.lookAt(CENTER);

    azimuthHandle.position.set(
      AZIMUTH_RADIUS * Math.sin(azRad),
      0.05,
      AZIMUTH_RADIUS * Math.cos(azRad)
    );
    elevationHandle.position.set(
      -0.8,
      ELEVATION_RADIUS * Math.sin(elRad) + CENTER.y,
      ELEVATION_RADIUS * Math.cos(elRad)
    );

    const orangeDist = distance - 0.5;
    distanceHandle.position.set(
      orangeDist * Math.sin(azRad) * Math.cos(elRad),
      orangeDist * Math.sin(elRad) + CENTER.y,
      orangeDist * Math.cos(azRad) * Math.cos(elRad)
    );
    distanceLineGeo.setFromPoints([cameraGroup.position.clone(), CENTER.clone()]);

    overlay.textContent = buildPrompt(azimuthAngle, elevationAngle, distanceFactor);
  }

  function syncWidgets() {
    const azSnap = snapToNearest(azimuthAngle, AZIMUTH_STEPS);
    const elSnap = snapToNearest(elevationAngle, ELEVATION_STEPS);
    const distSnap = snapToNearest(distanceFactor, DISTANCE_STEPS);
    writeWidgetValue(node, "azimuth", azSnap);
    writeWidgetValue(node, "elevation", elSnap);
    writeWidgetValue(node, "distance", distSnap);
  }

  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let isDragging = false;
  let dragTarget = null;
  let dragStartMouse = new THREE.Vector2();
  let dragStartDistance = 1.0;
  const intersection = new THREE.Vector3();

  const canvas = renderer.domElement;

  function updateMouse(event) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  function handlePointerDown(event) {
    if (event.preventDefault) {
      event.preventDefault();
    }
    if (event.stopPropagation) {
      event.stopPropagation();
    }
    updateMouse(event);
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects([
      azimuthHandle,
      elevationHandle,
      distanceHandle,
    ]);
    if (intersects.length > 0) {
      isDragging = true;
      dragTarget = intersects[0].object;
      dragTarget.material.emissiveIntensity = 1.0;
      dragTarget.scale.setScalar(1.3);
      dragStartMouse.copy(mouse);
      dragStartDistance = distanceFactor;
      canvas.style.cursor = "grabbing";
    }
  }

  function handlePointerMove(event) {
    updateMouse(event);
    if (isDragging && dragTarget) {
      raycaster.setFromCamera(mouse, camera);
      if (dragTarget.userData.type === "azimuth") {
        const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
        if (raycaster.ray.intersectPlane(plane, intersection)) {
          azimuthAngle = THREE.MathUtils.radToDeg(
            Math.atan2(intersection.x, intersection.z)
          );
          if (azimuthAngle < 0) {
            azimuthAngle += 360;
          }
        }
      } else if (dragTarget.userData.type === "elevation") {
        const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), -0.8);
        if (raycaster.ray.intersectPlane(plane, intersection)) {
          const relY = intersection.y - CENTER.y;
          const relZ = intersection.z;
          elevationAngle = THREE.MathUtils.clamp(
            THREE.MathUtils.radToDeg(Math.atan2(relY, relZ)),
            -30,
            60
          );
        }
      } else if (dragTarget.userData.type === "distance") {
        const deltaY = mouse.y - dragStartMouse.y;
        distanceFactor = THREE.MathUtils.clamp(
          dragStartDistance - deltaY * 1.5,
          0.6,
          1.8
        );
      }
      updatePositions();
    } else {
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects([
        azimuthHandle,
        elevationHandle,
        distanceHandle,
      ]);
      [azimuthHandle, elevationHandle, distanceHandle].forEach((handle) => {
        handle.material.emissiveIntensity = 0.5;
        handle.scale.setScalar(1);
      });
      if (intersects.length > 0) {
        intersects[0].object.material.emissiveIntensity = 0.8;
        intersects[0].object.scale.setScalar(1.1);
        canvas.style.cursor = "grab";
      } else {
        canvas.style.cursor = "default";
      }
    }
  }

  function handlePointerUp() {
    if (dragTarget) {
      dragTarget.material.emissiveIntensity = 0.5;
      dragTarget.scale.setScalar(1);

      const targetAz = snapToNearest(azimuthAngle, AZIMUTH_STEPS);
      const targetEl = snapToNearest(elevationAngle, ELEVATION_STEPS);
      const targetDist = snapToNearest(distanceFactor, DISTANCE_STEPS);

      const startAz = azimuthAngle;
      const startEl = elevationAngle;
      const startDist = distanceFactor;
      const startTime = Date.now();

      const animateSnap = () => {
        const t = Math.min((Date.now() - startTime) / 200, 1);
        const ease = 1 - Math.pow(1 - t, 3);

        let azDiff = targetAz - startAz;
        if (azDiff > 180) {
          azDiff -= 360;
        }
        if (azDiff < -180) {
          azDiff += 360;
        }
        azimuthAngle = startAz + azDiff * ease;
        if (azimuthAngle < 0) {
          azimuthAngle += 360;
        }
        if (azimuthAngle >= 360) {
          azimuthAngle -= 360;
        }

        elevationAngle = startEl + (targetEl - startEl) * ease;
        distanceFactor = startDist + (targetDist - startDist) * ease;

        updatePositions();
        if (t < 1) {
          requestAnimationFrame(animateSnap);
        } else {
          syncWidgets();
        }
      };
      animateSnap();
    }
    isDragging = false;
    dragTarget = null;
    canvas.style.cursor = "default";
  }

  canvas.addEventListener("mousedown", handlePointerDown);
  canvas.addEventListener("mousemove", handlePointerMove);
  canvas.addEventListener("mouseup", handlePointerUp);
  canvas.addEventListener("mouseleave", handlePointerUp);

  canvas.addEventListener(
    "touchstart",
    (event) => {
      event.preventDefault();
      if (event.touches.length) {
        handlePointerDown(event.touches[0]);
      }
    },
    { passive: false }
  );
  canvas.addEventListener(
    "touchmove",
    (event) => {
      event.preventDefault();
      if (event.touches.length) {
        handlePointerMove(event.touches[0]);
      }
    },
    { passive: false }
  );
  canvas.addEventListener(
    "touchend",
    (event) => {
      event.preventDefault();
      handlePointerUp();
    },
    { passive: false }
  );
  canvas.addEventListener(
    "touchcancel",
    (event) => {
      event.preventDefault();
      handlePointerUp();
    },
    { passive: false }
  );

  const resizeObserver = new ResizeObserver(() => {
    const width = wrapper.clientWidth || 1;
    const height = wrapper.clientHeight || 1;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  });
  resizeObserver.observe(wrapper);

  let disposed = false;
  function render() {
    if (disposed) {
      return;
    }
    requestAnimationFrame(render);
    renderer.render(scene, camera);
  }
  updatePositions();
  render();

  return {
    syncFromWidgets: () => {
      azimuthAngle = readWidgetValue(node, "azimuth", azimuthAngle);
      elevationAngle = readWidgetValue(node, "elevation", elevationAngle);
      distanceFactor = readWidgetValue(node, "distance", distanceFactor);
      updatePositions();
    },
    updateTexture: (base64) => {
      const normalized = Array.isArray(base64) ? base64[0] : base64;
      updateTextureFromBase64(normalized || "");
    },
    dispose: () => {
      disposed = true;
      resizeObserver.disconnect();
      renderer.dispose();
    },
  };
}

app.registerExtension({
  name: EXTENSION_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) {
      return;
    }

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      hideCameraWidgets(this);

      const element = document.createElement("div");
      const widget = this.addDOMWidget(NODE_NAME, "Camera3DWidget", element, {
        serialize: false,
        hideOnZoom: false,
      });
      this.setSize([DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT]);
      this.resizable = true;
      this._camera3dElement = element;
      this._camera3dHeight = Math.max(
        MIN_CAMERA_HEIGHT,
        this.size[1] - RESERVED_WIDGET_SPACE
      );
      element.style.height = `${this._camera3dHeight}px`;
      element.style.width = "100%";
      element.style.boxSizing = "border-box";
      widget.computeSize = (width) => [width, this._camera3dHeight];

      createCameraWidget(this, element).then((controller) => {
        this._camera3dController = controller;
        if (controller) {
          controller.syncFromWidgets();
        }
      });

      this._camera3dWidget = widget;
    });

    chainCallback(nodeType.prototype, "onConfigure", function () {
      hideCameraWidgets(this);
      if (this._camera3dController) {
        this._camera3dController.syncFromWidgets();
      }
    });

    chainCallback(nodeType.prototype, "onResize", function () {
      if (!this._camera3dElement || !this.size) {
        return;
      }
      const nextHeight = Math.max(
        MIN_CAMERA_HEIGHT,
        this.size[1] - RESERVED_WIDGET_SPACE
      );
      if (nextHeight !== this._camera3dHeight) {
        this._camera3dHeight = nextHeight;
        this._camera3dElement.style.height = `${this._camera3dHeight}px`;
        this.setDirtyCanvas(true, true);
      }
    });

    chainCallback(nodeType.prototype, "onExecuted", function (message) {
      if (!this._camera3dController || !message) {
        return;
      }
      if (message.bg_image) {
        this._camera3dController.updateTexture(message.bg_image);
      }
    });

    chainCallback(nodeType.prototype, "onRemoved", function () {
      if (this._camera3dController) {
        this._camera3dController.dispose();
        this._camera3dController = null;
      }
    });
  },
});
