import gradio as gr
import numpy as np
import random
import torch
import spaces

from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
#from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
#from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

MAX_SEED = np.iinfo(np.int32).max

# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=dtype
).to(device)

# Load the lightning LoRA for fast inference
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Edit-2511-Lightning",
    weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    adapter_name="lightning"
)

# Load the multi-angles LoRA
pipe.load_lora_weights(
    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors",
    adapter_name="angles"
)

pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])

# --- Prompt Building ---

# Azimuth mappings (8 positions)
AZIMUTH_MAP = {
    0: "front view",
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view"
}

# Elevation mappings (4 positions)
ELEVATION_MAP = {
    -30: "low-angle shot",
    0: "eye-level shot",
    30: "elevated shot",
    60: "high-angle shot"
}

# Distance mappings (3 positions)
DISTANCE_MAP = {
    0.6: "close-up",
    1.0: "medium shot",
    1.8: "wide shot"
}


def snap_to_nearest(value, options):
    """Snap a value to the nearest option in a list."""
    return min(options, key=lambda x: abs(x - value))


def build_camera_prompt(azimuth: float, elevation: float, distance: float) -> str:
    """
    Build a camera prompt from azimuth, elevation, and distance values.
    
    Args:
        azimuth: Horizontal rotation in degrees (0-360)
        elevation: Vertical angle in degrees (-30 to 60)
        distance: Distance factor (0.6 to 1.8)
    
    Returns:
        Formatted prompt string for the LoRA
    """
    # Snap to nearest valid values
    azimuth_snapped = snap_to_nearest(azimuth, list(AZIMUTH_MAP.keys()))
    elevation_snapped = snap_to_nearest(elevation, list(ELEVATION_MAP.keys()))
    distance_snapped = snap_to_nearest(distance, list(DISTANCE_MAP.keys()))
    
    azimuth_name = AZIMUTH_MAP[azimuth_snapped]
    elevation_name = ELEVATION_MAP[elevation_snapped]
    distance_name = DISTANCE_MAP[distance_snapped]
    
    return f"<sks> {azimuth_name} {elevation_name} {distance_name}"


@spaces.GPU
def infer_camera_edit(
    image: Image.Image,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    distance: float = 1.0,
    seed: int = 0,
    randomize_seed: bool = True,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 4,
    height: int = 1024,
    width: int = 1024,
):
    """
    Edit the camera angle of an image using Qwen Image Edit 2511 with multi-angles LoRA.
    """
    progress = gr.Progress(track_tqdm=True)
    
    prompt = build_camera_prompt(azimuth, elevation, distance)
    print(f"Generated Prompt: {prompt}")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    if image is None:
        raise gr.Error("Please upload an image first.")

    pil_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")

    result = pipe(
        image=[pil_image],
        prompt=prompt,
        height=height if height != 0 else None,
        width=width if width != 0 else None,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
    ).images[0]

    return result, seed, prompt


def update_dimensions_on_upload(image):
    """Compute recommended dimensions preserving aspect ratio."""
    if image is None:
        return 1024, 1024

    original_width, original_height = image.size

    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)

    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    return new_width, new_height


# --- 3D Camera Control Component ---
class CameraControl3D(gr.HTML):
    """
    A 3D camera control component using Three.js.
    Outputs: { azimuth: number, elevation: number, distance: number }
    Accepts imageUrl prop to display user's uploaded image on the plane.
    """
    def __init__(self, value=None, imageUrl=None, **kwargs):
        if value is None:
            value = {"azimuth": 0, "elevation": 0, "distance": 1.0}
        
        html_template = """
        <div id="camera-control-wrapper" style="width: 100%; height: 450px; position: relative; background: #1a1a1a; border-radius: 12px; overflow: hidden;">
            <div id="prompt-overlay" style="position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); padding: 8px 16px; border-radius: 8px; font-family: monospace; font-size: 12px; color: #00ff88; white-space: nowrap; z-index: 10;"></div>
        </div>
        """
        
        js_on_load = """
        (() => {
            const wrapper = element.querySelector('#camera-control-wrapper');
            const promptOverlay = element.querySelector('#prompt-overlay');
            
            // Wait for THREE to load
            const initScene = () => {
                if (typeof THREE === 'undefined') {
                    setTimeout(initScene, 100);
                    return;
                }
                
                // Scene setup
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a1a);
                
                const camera = new THREE.PerspectiveCamera(50, wrapper.clientWidth / wrapper.clientHeight, 0.1, 1000);
                camera.position.set(4.5, 3, 4.5);
                camera.lookAt(0, 0.75, 0);
                
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                wrapper.insertBefore(renderer.domElement, promptOverlay);
                
                // Lighting
                scene.add(new THREE.AmbientLight(0xffffff, 0.6));
                const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
                dirLight.position.set(5, 10, 5);
                scene.add(dirLight);
                
                // Grid
                scene.add(new THREE.GridHelper(8, 16, 0x333333, 0x222222));
                
                // Constants - reduced distances for tighter framing
                const CENTER = new THREE.Vector3(0, 0.75, 0);
                const BASE_DISTANCE = 1.6;
                const AZIMUTH_RADIUS = 2.4;
                const ELEVATION_RADIUS = 1.8;
                
                // State
                let azimuthAngle = props.value?.azimuth || 0;
                let elevationAngle = props.value?.elevation || 0;
                let distanceFactor = props.value?.distance || 1.0;
                
                // Mappings - reduced wide shot multiplier
                const azimuthSteps = [0, 45, 90, 135, 180, 225, 270, 315];
                const elevationSteps = [-30, 0, 30, 60];
                const distanceSteps = [0.6, 1.0, 1.4];
                
                const azimuthNames = {
                    0: 'front view', 45: 'front-right quarter view', 90: 'right side view',
                    135: 'back-right quarter view', 180: 'back view', 225: 'back-left quarter view',
                    270: 'left side view', 315: 'front-left quarter view'
                };
                const elevationNames = { '-30': 'low-angle shot', '0': 'eye-level shot', '30': 'elevated shot', '60': 'high-angle shot' };
                const distanceNames = { '0.6': 'close-up', '1': 'medium shot', '1.4': 'wide shot' };
                
                function snapToNearest(value, steps) {
                    return steps.reduce((prev, curr) => Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev);
                }
                
                // Create placeholder texture (smiley face)
                function createPlaceholderTexture() {
                    const canvas = document.createElement('canvas');
                    canvas.width = 256;
                    canvas.height = 256;
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#3a3a4a';
                    ctx.fillRect(0, 0, 256, 256);
                    ctx.fillStyle = '#ffcc99';
                    ctx.beginPath();
                    ctx.arc(128, 128, 80, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillStyle = '#333';
                    ctx.beginPath();
                    ctx.arc(100, 110, 10, 0, Math.PI * 2);
                    ctx.arc(156, 110, 10, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = '#333';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.arc(128, 130, 35, 0.2, Math.PI - 0.2);
                    ctx.stroke();
                    return new THREE.CanvasTexture(canvas);
                }
                
                // Target image plane
                let currentTexture = createPlaceholderTexture();
                const planeMaterial = new THREE.MeshBasicMaterial({ map: currentTexture, side: THREE.DoubleSide });
                let targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
                targetPlane.position.copy(CENTER);
                scene.add(targetPlane);
                
                // Function to update texture from image URL
                function updateTextureFromUrl(url) {
                    if (!url) {
                        // Reset to placeholder
                        planeMaterial.map = createPlaceholderTexture();
                        planeMaterial.needsUpdate = true;
                        // Reset plane to square
                        scene.remove(targetPlane);
                        targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
                        targetPlane.position.copy(CENTER);
                        scene.add(targetPlane);
                        return;
                    }
                    
                    const loader = new THREE.TextureLoader();
                    loader.crossOrigin = 'anonymous';
                    loader.load(url, (texture) => {
                        texture.minFilter = THREE.LinearFilter;
                        texture.magFilter = THREE.LinearFilter;
                        planeMaterial.map = texture;
                        planeMaterial.needsUpdate = true;
                        
                        // Adjust plane aspect ratio to match image
                        const img = texture.image;
                        if (img && img.width && img.height) {
                            const aspect = img.width / img.height;
                            const maxSize = 1.5;
                            let planeWidth, planeHeight;
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
                    }, undefined, (err) => {
                        console.error('Failed to load texture:', err);
                    });
                }
                
                // Check for initial imageUrl
                if (props.imageUrl) {
                    updateTextureFromUrl(props.imageUrl);
                }
                
                // Camera model
                const cameraGroup = new THREE.Group();
                const bodyMat = new THREE.MeshStandardMaterial({ color: 0x6699cc, metalness: 0.5, roughness: 0.3 });
                const body = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.22, 0.38), bodyMat);
                cameraGroup.add(body);
                const lens = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.09, 0.11, 0.18, 16),
                    new THREE.MeshStandardMaterial({ color: 0x6699cc, metalness: 0.5, roughness: 0.3 })
                );
                lens.rotation.x = Math.PI / 2;
                lens.position.z = 0.26;
                cameraGroup.add(lens);
                scene.add(cameraGroup);
                
                // GREEN: Azimuth ring
                const azimuthRing = new THREE.Mesh(
                    new THREE.TorusGeometry(AZIMUTH_RADIUS, 0.04, 16, 64),
                    new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.3 })
                );
                azimuthRing.rotation.x = Math.PI / 2;
                azimuthRing.position.y = 0.05;
                scene.add(azimuthRing);
                
                const azimuthHandle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.18, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.5 })
                );
                azimuthHandle.userData.type = 'azimuth';
                scene.add(azimuthHandle);
                
                // PINK: Elevation arc
                const arcPoints = [];
                for (let i = 0; i <= 32; i++) {
                    const angle = THREE.MathUtils.degToRad(-30 + (90 * i / 32));
                    arcPoints.push(new THREE.Vector3(-0.8, ELEVATION_RADIUS * Math.sin(angle) + CENTER.y, ELEVATION_RADIUS * Math.cos(angle)));
                }
                const arcCurve = new THREE.CatmullRomCurve3(arcPoints);
                const elevationArc = new THREE.Mesh(
                    new THREE.TubeGeometry(arcCurve, 32, 0.04, 8, false),
                    new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.3 })
                );
                scene.add(elevationArc);
                
                const elevationHandle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.18, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.5 })
                );
                elevationHandle.userData.type = 'elevation';
                scene.add(elevationHandle);
                
                // ORANGE: Distance line & handle
                const distanceLineGeo = new THREE.BufferGeometry();
                const distanceLine = new THREE.Line(distanceLineGeo, new THREE.LineBasicMaterial({ color: 0xffa500 }));
                scene.add(distanceLine);
                
                const distanceHandle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.18, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0xffa500, emissive: 0xffa500, emissiveIntensity: 0.5 })
                );
                distanceHandle.userData.type = 'distance';
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
                    
                    azimuthHandle.position.set(AZIMUTH_RADIUS * Math.sin(azRad), 0.05, AZIMUTH_RADIUS * Math.cos(azRad));
                    elevationHandle.position.set(-0.8, ELEVATION_RADIUS * Math.sin(elRad) + CENTER.y, ELEVATION_RADIUS * Math.cos(elRad));
                    
                    const orangeDist = distance - 0.5;
                    distanceHandle.position.set(
                        orangeDist * Math.sin(azRad) * Math.cos(elRad),
                        orangeDist * Math.sin(elRad) + CENTER.y,
                        orangeDist * Math.cos(azRad) * Math.cos(elRad)
                    );
                    distanceLineGeo.setFromPoints([cameraGroup.position.clone(), CENTER.clone()]);
                    
                    // Update prompt
                    const azSnap = snapToNearest(azimuthAngle, azimuthSteps);
                    const elSnap = snapToNearest(elevationAngle, elevationSteps);
                    const distSnap = snapToNearest(distanceFactor, distanceSteps);
                    const distKey = distSnap === 1 ? '1' : distSnap.toFixed(1);
                    const prompt = '<sks> ' + azimuthNames[azSnap] + ' ' + elevationNames[String(elSnap)] + ' ' + distanceNames[distKey];
                    promptOverlay.textContent = prompt;
                }
                
                function updatePropsAndTrigger() {
                    const azSnap = snapToNearest(azimuthAngle, azimuthSteps);
                    const elSnap = snapToNearest(elevationAngle, elevationSteps);
                    const distSnap = snapToNearest(distanceFactor, distanceSteps);
                    
                    props.value = { azimuth: azSnap, elevation: elSnap, distance: distSnap };
                    trigger('change', props.value);
                }
                
                // Raycasting
                const raycaster = new THREE.Raycaster();
                const mouse = new THREE.Vector2();
                let isDragging = false;
                let dragTarget = null;
                let dragStartMouse = new THREE.Vector2();
                let dragStartDistance = 1.0;
                const intersection = new THREE.Vector3();
                
                const canvas = renderer.domElement;
                
                canvas.addEventListener('mousedown', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    raycaster.setFromCamera(mouse, camera);
                    const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle, distanceHandle]);
                    
                    if (intersects.length > 0) {
                        isDragging = true;
                        dragTarget = intersects[0].object;
                        dragTarget.material.emissiveIntensity = 1.0;
                        dragTarget.scale.setScalar(1.3);
                        dragStartMouse.copy(mouse);
                        dragStartDistance = distanceFactor;
                        canvas.style.cursor = 'grabbing';
                    }
                });
                
                canvas.addEventListener('mousemove', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    if (isDragging && dragTarget) {
                        raycaster.setFromCamera(mouse, camera);
                        
                        if (dragTarget.userData.type === 'azimuth') {
                            const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                azimuthAngle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                                if (azimuthAngle < 0) azimuthAngle += 360;
                            }
                        } else if (dragTarget.userData.type === 'elevation') {
                            const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), -0.8);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                const relY = intersection.y - CENTER.y;
                                const relZ = intersection.z;
                                elevationAngle = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(Math.atan2(relY, relZ)), -30, 60);
                            }
                        } else if (dragTarget.userData.type === 'distance') {
                            const deltaY = mouse.y - dragStartMouse.y;
                            distanceFactor = THREE.MathUtils.clamp(dragStartDistance - deltaY * 1.5, 0.6, 1.4);
                        }
                        updatePositions();
                    } else {
                        raycaster.setFromCamera(mouse, camera);
                        const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle, distanceHandle]);
                        [azimuthHandle, elevationHandle, distanceHandle].forEach(h => {
                            h.material.emissiveIntensity = 0.5;
                            h.scale.setScalar(1);
                        });
                        if (intersects.length > 0) {
                            intersects[0].object.material.emissiveIntensity = 0.8;
                            intersects[0].object.scale.setScalar(1.1);
                            canvas.style.cursor = 'grab';
                        } else {
                            canvas.style.cursor = 'default';
                        }
                    }
                });
                
                const onMouseUp = () => {
                    if (dragTarget) {
                        dragTarget.material.emissiveIntensity = 0.5;
                        dragTarget.scale.setScalar(1);
                        
                        // Snap and animate
                        const targetAz = snapToNearest(azimuthAngle, azimuthSteps);
                        const targetEl = snapToNearest(elevationAngle, elevationSteps);
                        const targetDist = snapToNearest(distanceFactor, distanceSteps);
                        
                        const startAz = azimuthAngle, startEl = elevationAngle, startDist = distanceFactor;
                        const startTime = Date.now();
                        
                        function animateSnap() {
                            const t = Math.min((Date.now() - startTime) / 200, 1);
                            const ease = 1 - Math.pow(1 - t, 3);
                            
                            let azDiff = targetAz - startAz;
                            if (azDiff > 180) azDiff -= 360;
                            if (azDiff < -180) azDiff += 360;
                            azimuthAngle = startAz + azDiff * ease;
                            if (azimuthAngle < 0) azimuthAngle += 360;
                            if (azimuthAngle >= 360) azimuthAngle -= 360;
                            
                            elevationAngle = startEl + (targetEl - startEl) * ease;
                            distanceFactor = startDist + (targetDist - startDist) * ease;
                            
                            updatePositions();
                            if (t < 1) requestAnimationFrame(animateSnap);
                            else updatePropsAndTrigger();
                        }
                        animateSnap();
                    }
                    isDragging = false;
                    dragTarget = null;
                    canvas.style.cursor = 'default';
                };
                
                canvas.addEventListener('mouseup', onMouseUp);
                canvas.addEventListener('mouseleave', onMouseUp);

                // Touch support for mobile
                canvas.addEventListener('touchstart', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    raycaster.setFromCamera(mouse, camera);
                    const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle, distanceHandle]);
                    
                    if (intersects.length > 0) {
                        isDragging = true;
                        dragTarget = intersects[0].object;
                        dragTarget.material.emissiveIntensity = 1.0;
                        dragTarget.scale.setScalar(1.3);
                        dragStartMouse.copy(mouse);
                        dragStartDistance = distanceFactor;
                    }
                }, { passive: false });
                
                canvas.addEventListener('touchmove', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    if (isDragging && dragTarget) {
                        raycaster.setFromCamera(mouse, camera);
                        
                        if (dragTarget.userData.type === 'azimuth') {
                            const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                azimuthAngle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                                if (azimuthAngle < 0) azimuthAngle += 360;
                            }
                        } else if (dragTarget.userData.type === 'elevation') {
                            const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), -0.8);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                const relY = intersection.y - CENTER.y;
                                const relZ = intersection.z;
                                elevationAngle = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(Math.atan2(relY, relZ)), -30, 60);
                            }
                        } else if (dragTarget.userData.type === 'distance') {
                            const deltaY = mouse.y - dragStartMouse.y;
                            distanceFactor = THREE.MathUtils.clamp(dragStartDistance - deltaY * 1.5, 0.6, 1.4);
                        }
                        updatePositions();
                    }
                }, { passive: false });
                
                canvas.addEventListener('touchend', (e) => {
                    e.preventDefault();
                    onMouseUp();
                }, { passive: false });
                
                canvas.addEventListener('touchcancel', (e) => {
                    e.preventDefault();
                    onMouseUp();
                }, { passive: false });
                
                // Initial update
                updatePositions();
                
                // Render loop
                function render() {
                    requestAnimationFrame(render);
                    renderer.render(scene, camera);
                }
                render();
                
                // Handle resize
                new ResizeObserver(() => {
                    camera.aspect = wrapper.clientWidth / wrapper.clientHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
                }).observe(wrapper);
                
                // Store update functions for external calls
                wrapper._updateFromProps = (newVal) => {
                    if (newVal && typeof newVal === 'object') {
                        azimuthAngle = newVal.azimuth ?? azimuthAngle;
                        elevationAngle = newVal.elevation ?? elevationAngle;
                        distanceFactor = newVal.distance ?? distanceFactor;
                        updatePositions();
                    }
                };
                
                wrapper._updateTexture = updateTextureFromUrl;
                
                // Watch for prop changes (imageUrl and value)
                let lastImageUrl = props.imageUrl;
                let lastValue = JSON.stringify(props.value);
                setInterval(() => {
                    // Check imageUrl changes
                    if (props.imageUrl !== lastImageUrl) {
                        lastImageUrl = props.imageUrl;
                        updateTextureFromUrl(props.imageUrl);
                    }
                    // Check value changes (from sliders)
                    const currentValue = JSON.stringify(props.value);
                    if (currentValue !== lastValue) {
                        lastValue = currentValue;
                        if (props.value && typeof props.value === 'object') {
                            azimuthAngle = props.value.azimuth ?? azimuthAngle;
                            elevationAngle = props.value.elevation ?? elevationAngle;
                            distanceFactor = props.value.distance ?? distanceFactor;
                            updatePositions();
                        }
                    }
                }, 100);
            };
            
            initScene();
        })();
        """
        
        super().__init__(
            value=value,
            html_template=html_template,
            js_on_load=js_on_load,
            imageUrl=imageUrl,
            **kwargs
        )


# --- UI ---
css = '''
#col-container { max-width: 1200px; margin: 0 auto; }
.dark .progress-text { color: white !important; }
#camera-3d-control { min-height: 450px; }
.slider-row { display: flex; gap: 10px; align-items: center; }
'''

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 Qwen Image Edit 2511 — 3D Camera Control
    
    Control camera angles using the **3D viewport** or **sliders**. 
    Using [fal's Qwen-Image-Edit-2511-Multiple-Angles-LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) for precise camera control.
    """)
    
    with gr.Row():
        # Left column: Input image and controls
        with gr.Column(scale=1):
            image = gr.Image(label="Input Image", type="pil", height=300)
            
            gr.Markdown("### 🎮 3D Camera Control")
            gr.Markdown("*Drag the colored handles: 🟢 Azimuth, 🩷 Elevation, 🟠 Distance*")
            
            camera_3d = CameraControl3D(
                value={"azimuth": 0, "elevation": 0, "distance": 1.0},
                elem_id="camera-3d-control"
            )

            run_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
            
            gr.Markdown("### 🎚️ Slider Controls")
            
            azimuth_slider = gr.Slider(
                label="Azimuth (Horizontal Rotation)",
                minimum=0,
                maximum=315,
                step=45,
                value=0,
                info="0°=front, 90°=right, 180°=back, 270°=left"
            )
            
            elevation_slider = gr.Slider(
                label="Elevation (Vertical Angle)", 
                minimum=-30,
                maximum=60,
                step=30,
                value=0,
                info="-30°=low angle, 0°=eye level, 60°=high angle"
            )
            
            distance_slider = gr.Slider(
                label="Distance",
                minimum=0.6,
                maximum=1.4,
                step=0.4,
                value=1.0,
                info="0.6=close-up, 1.0=medium, 1.4=wide"
            )
            
            prompt_preview = gr.Textbox(
                label="Generated Prompt",
                value="<sks> front view eye-level shot medium shot",
                interactive=False
            )
        
        # Right column: Output
        with gr.Column(scale=1):
            result = gr.Image(label="Output Image", height=500)
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=20, step=1, value=4)
                height = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=1024)
                width = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=1024)
    
    # --- Event Handlers ---
    
    def update_prompt_from_sliders(azimuth, elevation, distance):
        """Update prompt preview when sliders change."""
        prompt = build_camera_prompt(azimuth, elevation, distance)
        return prompt
    
    def sync_3d_to_sliders(camera_value):
        """Sync 3D control changes to sliders."""
        if camera_value and isinstance(camera_value, dict):
            az = camera_value.get('azimuth', 0)
            el = camera_value.get('elevation', 0)
            dist = camera_value.get('distance', 1.0)
            prompt = build_camera_prompt(az, el, dist)
            return az, el, dist, prompt
        return gr.update(), gr.update(), gr.update(), gr.update()
    
    def sync_sliders_to_3d(azimuth, elevation, distance):
        """Sync slider changes to 3D control."""
        return {"azimuth": azimuth, "elevation": elevation, "distance": distance}
    
    def update_3d_image(image):
        """Update the 3D component with the uploaded image."""
        if image is None:
            return gr.update(imageUrl=None)
        # Convert PIL image to base64 data URL
        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_str}"
        return gr.update(imageUrl=data_url)
    
    # Slider -> Prompt preview
    for slider in [azimuth_slider, elevation_slider, distance_slider]:
        slider.change(
            fn=update_prompt_from_sliders,
            inputs=[azimuth_slider, elevation_slider, distance_slider],
            outputs=[prompt_preview]
        )
    
    # 3D control -> Sliders + Prompt
    camera_3d.change(
        fn=sync_3d_to_sliders,
        inputs=[camera_3d],
        outputs=[azimuth_slider, elevation_slider, distance_slider, prompt_preview]
    )
    
    # Sliders -> 3D control
    for slider in [azimuth_slider, elevation_slider, distance_slider]:
        slider.release(
            fn=sync_sliders_to_3d,
            inputs=[azimuth_slider, elevation_slider, distance_slider],
            outputs=[camera_3d]
        )
    
    # Generate button
    run_btn.click(
        fn=infer_camera_edit,
        inputs=[image, azimuth_slider, elevation_slider, distance_slider, seed, randomize_seed, guidance_scale, num_inference_steps, height, width],
        outputs=[result, seed, prompt_preview]
    )
    
    # Image upload -> update dimensions AND update 3D preview
    image.upload(
        fn=update_dimensions_on_upload,
        inputs=[image],
        outputs=[width, height]
    ).then(
        fn=update_3d_image,
        inputs=[image],
        outputs=[camera_3d]
    )
    
    # Also handle image clear
    image.clear(
        fn=lambda: gr.update(imageUrl=None),
        outputs=[camera_3d]
    )
    
    # Examples
    # gr.Examples(
    #    examples=[
    #        ["example1.jpg", 90, 0, 1.0],
    #        ["example2.jpg", 0, 30, 0.6],
    #        ["example3.jpg", 180, -30, 1.8],
    #    ],
    #    inputs=[image, azimuth_slider, elevation_slider, distance_slider],
    #    outputs=[result, seed, prompt_preview],
    #    fn=lambda img, az, el, dist: infer_camera_edit(img, az, el, dist),
    #    cache_examples=False,
    #)

if __name__ == "__main__":
    head = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    css = '.fillable{max-width: 1200px !important}'
    demo.launch(head=head, css=css)