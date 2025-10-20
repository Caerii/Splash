import { Camera } from './camera'
import { mlsmpmParticleStructSize, MLSMPMSimulator } from './mls-mpm/mls-mpm'
import { renderUniformsViews, renderUniformsValues, numParticlesMax, initCommonGlobals } from './common'
import { FluidRenderer } from './render/fluidRender'
import GUI from 'lil-gui';

// New modular imports
import { SIMULATION, PARTICLE_OPTIONS, SHAPE_OPTIONS, GUI as GUI_CONSTANTS, BUFFER_SIZES } from './utils/Constants'
import { BufferManager } from './buffers/BufferManager'
import { UniformManager } from './buffers/UniformManager'
import { ShaderManager } from './shaders/ShaderManager'
import { PhysicsEngine } from './physics/PhysicsEngine'
import { ParameterManager } from './ui/ParameterManager'

/// <reference types="@webgpu/types" />


async function init() {
    const canvas: HTMLCanvasElement = document.querySelector('canvas')!

    if (!navigator.gpu) {
        alert("WebGPU is not supported on your browser.");
        throw new Error()
    }

    const adapter = await navigator.gpu.requestAdapter()

    if (!adapter) {
        alert("Adapter is not available.");
        throw new Error()
    }

    const device = await adapter.requestDevice()

    const context = canvas.getContext('webgpu') as GPUCanvasContext

    if (!context) {
        throw new Error()   
    }

    let devicePixelRatio  = 0.7;
    canvas.width = devicePixelRatio * canvas.clientWidth
    canvas.height = devicePixelRatio * canvas.clientHeight

    console.log(canvas.width, canvas.height)

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat()

    context.configure({
        device,
        format: presentationFormat,
    })

    return { canvas, device, presentationFormat, context }
}

async function main() {
    const { canvas, device, presentationFormat, context } = await init();
    
    console.log("initialization done")

    context.configure({
        device,
        format: presentationFormat,
    })

    let cubemapTexture: GPUTexture;
    {
        // The order of the array layers is [+X, -X, +Y, -Y, +Z, -Z]
        const imgSrcs = [
            'cubemap/posx.png',
            'cubemap/negx.png',
            'cubemap/posy.png',
            'cubemap/negy.png',
            'cubemap/posz.png',
            'cubemap/negz.png',
        ];
        const promises = imgSrcs.map(async (src) => {
            const response = await fetch(src);
            return createImageBitmap(await response.blob());
        });
        const imageBitmaps = await Promise.all(promises);

        cubemapTexture = device.createTexture({
            dimension: '2d',
            // Create a 2d array texture.
            // Assume each image has the same size.
            size: [imageBitmaps[0].width, imageBitmaps[0].height, 6],
            format: 'rgba8unorm',
            usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
        });

        for (let i = 0; i < imageBitmaps.length; i++) {
            const imageBitmap = imageBitmaps[i];
            device.queue.copyExternalImageToTexture(
                { source: imageBitmap },
                { texture: cubemapTexture, origin: [0, 0, i] },
                [imageBitmap.width, imageBitmap.height]
            );
        }
    }

    const cubemapTextureView = cubemapTexture.createView({
        dimension: 'cube',
    });
    console.log("cubemap initialization done")


    const gui = new GUI();

    // Helper function for camera positioning based on shape type
    function updateCameraPosition(shapeType: number, containerSize: number[], sphereRadius?: number) {
        let cameraDistance: number;
        let cameraTarget: number[];
        
        if (shapeType === 2) { // Sphere
            // For spheres, use the sphere radius for camera distance, but target container center
            const radius = sphereRadius || 25;
            cameraDistance = Math.max(radius * 4, 200); // Reasonable distance for sphere viewing
            cameraTarget = [containerSize[0] / 2, containerSize[1] / 2, containerSize[2] / 2];
        } else {
            const containerDiagonal = Math.sqrt(containerSize[0] * containerSize[0] + containerSize[1] * containerSize[1] + containerSize[2] * containerSize[2])
            cameraDistance = Math.max(containerDiagonal * 1.2, 200)
            cameraTarget = [containerSize[0] / 2, containerSize[1] / 2, containerSize[2] / 2]
        }
        camera.reset(cameraDistance, cameraTarget, mlsmpmFov, mlsmpmZoomRate)
    }

    // Initialize modular components (keeping only those we'll activate in Tier 2)
    const parameterManager = new ParameterManager();
    const uniformManager = new UniformManager(device, new BufferManager(device));
    const physicsEngine = new PhysicsEngine(uniformManager);
    
    // Initialize common.ts migration singleton
    initCommonGlobals(uniformManager);

    // Get options from constants
    const numParticlesOptions = PARTICLE_OPTIONS.map(opt => `${opt.label} (${opt.value.toLocaleString()} particles)`);
    const shapeOptions = SHAPE_OPTIONS;

    // Get parameters from ParameterManager
    const params = parameterManager.getParameters();
    
    // Set numParticles to the display string for GUI compatibility
    params.numParticles = numParticlesOptions[1]; // Default to Medium
    
    // Create shape type mapping
    const shapeTypeMap: { [key: string]: number } = SHAPE_OPTIONS.reduce((acc, shape, index) => {
        acc[shape] = index;
        return acc;
    }, {} as { [key: string]: number });
    
    // Add toggle functions to params for GUI compatibility
    params.toggleSimulation = () => {
        parameterManager.toggleSimulation();
        params.running = parameterManager.getParameter('running');
    };
    params.toggleFlipGravity = () => {
        parameterManager.toggleFlipGravity();
        params.flipGravity = parameterManager.getParameter('flipGravity');
    };  

    const numParticlesFolder = gui.addFolder(GUI_CONSTANTS.FOLDER_NAMES.PARTICLES);
    numParticlesFolder.add(params, 'numParticles', numParticlesOptions)
        .name('Number of Particles')
    
    const speedFolder = gui.addFolder(GUI_CONSTANTS.FOLDER_NAMES.SPEED);
    speedFolder.add(params, 'speed', 0.3, 1.0, 0.1).name('Simulation Speed')
    
    const colorFolder = gui.addFolder(GUI_CONSTANTS.FOLDER_NAMES.COLORS);
    colorFolder.add(params, 'r', 0, 255, 1).name('R')
    colorFolder.add(params, 'g', 0, 255, 1).name('G')
    colorFolder.add(params, 'b', 0, 255, 1).name('B')
    colorFolder.add(params, 'colorDensity', 0.0, 6.0, 0.1).name('Density')
    colorFolder.close();
    
    const physicsFolder = gui.addFolder(GUI_CONSTANTS.FOLDER_NAMES.PHYSICS);
    physicsFolder.add(params, 'gravity', GUI_CONSTANTS.CONTROL_RANGES.GRAVITY.min, GUI_CONSTANTS.CONTROL_RANGES.GRAVITY.max, GUI_CONSTANTS.CONTROL_RANGES.GRAVITY.step).name('Gravity')
    physicsFolder.add(params, 'particleSize', GUI_CONSTANTS.CONTROL_RANGES.PARTICLE_SIZE.min, GUI_CONSTANTS.CONTROL_RANGES.PARTICLE_SIZE.max, GUI_CONSTANTS.CONTROL_RANGES.PARTICLE_SIZE.step).name('Particle Size')
    physicsFolder.add(params, 'flipGravity').name('Flip Gravity')
    physicsFolder.add(params, 'flipSpeed', 0.1, 2.0, 0.1).name('Flip Speed')
    const viscosityControl = physicsFolder.add(params, 'viscosity', GUI_CONSTANTS.CONTROL_RANGES.VISCOSITY.min, GUI_CONSTANTS.CONTROL_RANGES.VISCOSITY.max, GUI_CONSTANTS.CONTROL_RANGES.VISCOSITY.step).name('Viscosity')
    const wallStiffnessControl = physicsFolder.add(params, 'wallStiffness', GUI_CONSTANTS.CONTROL_RANGES.WALL_STIFFNESS.min, GUI_CONSTANTS.CONTROL_RANGES.WALL_STIFFNESS.max, GUI_CONSTANTS.CONTROL_RANGES.WALL_STIFFNESS.step).name('Wall Stiffness')
    const collisionDampingControl = physicsFolder.add(params, 'collisionDamping', GUI_CONSTANTS.CONTROL_RANGES.COLLISION_DAMPING.min, GUI_CONSTANTS.CONTROL_RANGES.COLLISION_DAMPING.max, GUI_CONSTANTS.CONTROL_RANGES.COLLISION_DAMPING.step).name('Collision Damping')
    const velocityCapControl = physicsFolder.add(params, 'velocityCap', GUI_CONSTANTS.CONTROL_RANGES.VELOCITY_CAP.min, GUI_CONSTANTS.CONTROL_RANGES.VELOCITY_CAP.max, GUI_CONSTANTS.CONTROL_RANGES.VELOCITY_CAP.step).name('Velocity Cap')
    physicsFolder.close();

    // Physics controls are handled by the consolidated updatePhysicsProperties function below

    const shapeFolder = gui.addFolder(GUI_CONSTANTS.FOLDER_NAMES.SHAPES);
    const shapeControl = shapeFolder.add(params, 'shapeType', shapeOptions).name('Shape Type')
    shapeControl.onChange((value: string) => {
        // Calculate required container size for the new shape type
        const containerBoxSize = mlsmpmSimulator.calculateRequiredBoxSize(shapeTypeMap[value] || 0, params, initBoxSize)
        initBoxSize = containerBoxSize
        realBoxSize = [...initBoxSize]
        mlsmpmSimulator.changeBoxSize(realBoxSize)
        // Reset simulation with new shape type and properly sized container
        mlsmpmSimulator.reset(initBoxSize, mlsmpmNumParticleParams[paramsIdx], shapeTypeMap[value] || 0, params)
        // Update camera position based on shape type
        updateCameraPosition(shapeTypeMap[value] || 0, initBoxSize, params.sphereRadius)
    })

    // Physics properties change listeners - now use PhysicsEngine as authority
    const updatePhysicsProperties = () => {
        physicsEngine.setViscosity(params.viscosity);
        physicsEngine.setWallStiffness(params.wallStiffness);
        physicsEngine.setCollisionDamping(params.collisionDamping);
        physicsEngine.setVelocityCap(params.velocityCap);
        // PhysicsEngine now updates the simulator through UniformManager
        mlsmpmSimulator.updatePhysicsProperties(params.viscosity, params.wallStiffness, params.collisionDamping, params.velocityCap);
    };

    // Consolidated shape parameter change handler
    const handleShapeParameterChange = () => {
        mlsmpmSimulator.updateShapeParameters(params);
        const containerBoxSize = mlsmpmSimulator.calculateRequiredBoxSize(shapeTypeMap[params.shapeType] || 0, params, initBoxSize);
        initBoxSize = containerBoxSize;
        realBoxSize = [...initBoxSize];
        mlsmpmSimulator.changeBoxSize(realBoxSize);
        mlsmpmSimulator.reset(initBoxSize, mlsmpmNumParticleParams[paramsIdx], shapeTypeMap[params.shapeType] || 0, params);
        
        // Update camera position based on new container size
        const containerDiagonal = Math.sqrt(initBoxSize[0] * initBoxSize[0] + initBoxSize[1] * initBoxSize[1] + initBoxSize[2] * initBoxSize[2]);
        const cameraDistance = Math.max(containerDiagonal * 1.2, 200);
        const containerCenter = [initBoxSize[0] / 2, initBoxSize[1] / 2, initBoxSize[2] / 2];
        camera.reset(cameraDistance, containerCenter, mlsmpmFov, mlsmpmZoomRate);
    };

    // Specialized handler for sphere parameters (uses updateCameraPosition)
    const handleSphereParameterChange = () => {
        mlsmpmSimulator.updateShapeParameters(params);
        const containerBoxSize = mlsmpmSimulator.calculateRequiredBoxSize(shapeTypeMap[params.shapeType] || 0, params, initBoxSize);
        initBoxSize = containerBoxSize;
        realBoxSize = [...initBoxSize];
        mlsmpmSimulator.changeBoxSize(realBoxSize);
        mlsmpmSimulator.reset(initBoxSize, mlsmpmNumParticleParams[paramsIdx], shapeTypeMap[params.shapeType] || 0, params);
        // Update camera position based on sphere radius
        updateCameraPosition(2, initBoxSize, params.sphereRadius);
    };
    
    viscosityControl.onChange(updatePhysicsProperties);
    wallStiffnessControl.onChange(updatePhysicsProperties);
    collisionDampingControl.onChange(updatePhysicsProperties);
    velocityCapControl.onChange(updatePhysicsProperties);

    
    // Box controls
    const boxFolder = shapeFolder.addFolder('Box Parameters');
    const boxWidthControl = boxFolder.add(params, 'boxWidth', 20, 100, 1).name('Width')
    const boxHeightControl = boxFolder.add(params, 'boxHeight', 20, 80, 1).name('Height')
    const boxDepthControl = boxFolder.add(params, 'boxDepth', 20, 100, 1).name('Depth')
    boxWidthControl.onChange(handleShapeParameterChange)
    boxHeightControl.onChange(handleShapeParameterChange)
    boxDepthControl.onChange(handleShapeParameterChange)
    boxFolder.close();
    
    // Cylinder controls
    const cylinderFolder = shapeFolder.addFolder('Cylinder Parameters');
    const cylinderRadiusControl = cylinderFolder.add(params, 'cylinderRadius', 10, 50, 1).name('Radius')
    const cylinderHeightControl = cylinderFolder.add(params, 'cylinderHeight', 20, 80, 1).name('Height')
    cylinderRadiusControl.onChange(handleShapeParameterChange)
    cylinderHeightControl.onChange(handleShapeParameterChange)
    cylinderFolder.close();
    
    // Sphere controls
    const sphereFolder = shapeFolder.addFolder('Sphere Parameters');
    const sphereRadiusControl = sphereFolder.add(params, 'sphereRadius', 10, 40, 1).name('Radius')
    sphereRadiusControl.onChange(handleSphereParameterChange)
    sphereFolder.close();
    
    // Cone controls
    const coneFolder = shapeFolder.addFolder('Cone Parameters');
    const coneRadiusControl = coneFolder.add(params, 'coneRadius', 10, 50, 1).name('Base Radius')
    const coneHeightControl = coneFolder.add(params, 'coneHeight', 20, 80, 1).name('Height')
    const coneTaperControl = coneFolder.add(params, 'coneTaper', 0.0, 1.0, 0.01).name('Taper')
    coneRadiusControl.onChange(handleShapeParameterChange)
    coneHeightControl.onChange(handleShapeParameterChange)
    coneTaperControl.onChange(() => {
        mlsmpmSimulator.updateShapeParameters(params)
    })
    coneFolder.close();
    
    // Venturi controls
    const venturiFolder = shapeFolder.addFolder('Venturi Parameters');
    const venturiTopRadiusControl = venturiFolder.add(params, 'venturiTopRadius', 3, 60, 1).name('Top Radius')
    const venturiThroatRadiusControl = venturiFolder.add(params, 'venturiThroatRadius', 3, 60, 1).name('Throat Radius')
    const venturiBottomRadiusControl = venturiFolder.add(params, 'venturiBottomRadius', 3, 60, 1).name('Bottom Radius')
    const venturiHeightControl = venturiFolder.add(params, 'venturiHeight', 30, 100, 1).name('Height')
    const venturiThroatPositionControl = venturiFolder.add(params, 'venturiThroatPosition', 0.2, 0.8, 0.01).name('Throat Position')
    const venturiHelixCountControl = venturiFolder.add(params, 'venturiHelixCount', 2, 5, 1).name('Helix Count')
    const venturiHelixPitchControl = venturiFolder.add(params, 'venturiHelixPitch', 0.1, 0.8, 0.01).name('Helix Pitch')
    
    venturiTopRadiusControl.onChange(handleShapeParameterChange)
    venturiThroatRadiusControl.onChange(handleShapeParameterChange)
    venturiBottomRadiusControl.onChange(handleShapeParameterChange)
    venturiHeightControl.onChange(handleShapeParameterChange)
    venturiThroatPositionControl.onChange(handleShapeParameterChange)
    venturiHelixCountControl.onChange(handleShapeParameterChange)
    venturiHelixPitchControl.onChange(handleShapeParameterChange)
    venturiFolder.close();
    
    // Torus controls
    const torusFolder = shapeFolder.addFolder('Torus Parameters');
    const torusMajorRadiusControl = torusFolder.add(params, 'torusMajorRadius', 5, 60, 1).name('Major Radius')
    const torusMinorRadiusControl = torusFolder.add(params, 'torusMinorRadius', 1, 30, 1).name('Minor Radius')
    const torusHeightControl = torusFolder.add(params, 'torusHeight', 10, 120, 1).name('Height')
    const torusKnotCountControl = torusFolder.add(params, 'torusKnotCount', 0, 8, 1).name('Knot Count')
    const torusKnotIntensityControl = torusFolder.add(params, 'torusKnotIntensity', 0.0, 3.0, 0.01).name('Knot Intensity')
    const torusTwistAmountControl = torusFolder.add(params, 'torusTwistAmount', 0.0, 10.0, 0.01).name('Twist Amount')
    
    torusMajorRadiusControl.onChange(handleShapeParameterChange)
    torusMinorRadiusControl.onChange(handleShapeParameterChange)
    torusHeightControl.onChange(handleShapeParameterChange)
    torusKnotCountControl.onChange(handleShapeParameterChange)
    torusKnotIntensityControl.onChange(handleShapeParameterChange)
    torusTwistAmountControl.onChange(handleShapeParameterChange)
    torusFolder.close();
    
    // Chemotaxis controls
    const chemotaxisFolder = gui.addFolder('Chemotaxis');
    const chemotaxisEnabledControl = chemotaxisFolder.add(params, 'chemotaxisEnabled').name('Enabled')
    const chemotaxisStrengthControl = chemotaxisFolder.add(params, 'chemotaxisStrength', 0.0, 10.0, 0.01).name('Strength')
    const chemotaxisDiffusionRateControl = chemotaxisFolder.add(params, 'chemotaxisDiffusionRate', 0.0, 5.0, 0.01).name('Diffusion Rate')
    const chemotaxisDecayRateControl = chemotaxisFolder.add(params, 'chemotaxisDecayRate', 0.0, 2.0, 0.01).name('Decay Rate')
    const chemotaxisSourceRadiusControl = chemotaxisFolder.add(params, 'chemotaxisSourceRadius', 0.1, 200.0, 0.1).name('Source Radius')
    const chemotaxisAttractionControl = chemotaxisFolder.add(params, 'chemotaxisAttraction', -5.0, 5.0, 0.01).name('Attraction')
    
    chemotaxisEnabledControl.onChange(handleShapeParameterChange)
    chemotaxisStrengthControl.onChange(handleShapeParameterChange)
    chemotaxisDiffusionRateControl.onChange(handleShapeParameterChange)
    chemotaxisDecayRateControl.onChange(handleShapeParameterChange)
    chemotaxisSourceRadiusControl.onChange(handleShapeParameterChange)
    chemotaxisAttractionControl.onChange(handleShapeParameterChange)
    chemotaxisFolder.close();
    
    shapeFolder.close();

    document.addEventListener('keydown', (event) => {
        if (event.code === 'KeyP') { 
          params.toggleSimulation?.(); 
        }
        if (event.code === 'KeyT') { 
          params.toggleFlipGravity?.(); 
        }
    });



    renderUniformsViews.texel_size.set([1.0 / canvas.width, 1.0 / canvas.height]);

    // シミュレーションとレンダリングで使いまわすバッファ
    const maxParticleStructSize = mlsmpmParticleStructSize
    const maxGridCount = 140 * 140 * 140; // 2,744,000 grid cells to support largest box size
    const particleBuffer = device.createBuffer({
        label: 'particles buffer', 
        size: maxParticleStructSize * numParticlesMax, 
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const densityGridBuffer = device.createBuffer({
        label: 'density grid buffer', 
        size: 4 * maxGridCount, 
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const posvelBuffer = device.createBuffer({
        label: 'posvel buffer', 
        size: 32 * numParticlesMax,  
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const renderUniformBuffer = device.createBuffer({
        label: 'filter uniform buffer', 
        size: renderUniformsValues.byteLength, 
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const initBoxSizeBuffer = device.createBuffer({
        label: 'init box size buffer', 
        size: 12,  // vec3f
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const shapeParamsBuffer = device.createBuffer({
        label: 'shape parameters buffer', 
        size: BUFFER_SIZES.SHAPE_PARAMS,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    console.log("buffer allocating done")

    // Initialize shape parameters buffer with default values
    device.queue.writeBuffer(shapeParamsBuffer, 0, uniformManager.shapeParamsValues);

    let mlsmpmNumParticleParams = [40000, 70000, 100000, 600000, 500000, 1000000]
    let mlsmpmInitBoxSizes = [[60, 50, 60], [70, 50, 70], [80, 70, 80], [120, 120, 120], [130, 130, 130], [140, 140, 140]]
    let mlsmpmInitDistances = [200, 250, 300, 500, 600, 700]
    let mouseRadiuses = [15, 15, 15, 18, 20, 22]
    let cameraTargetY = [10, 12, 12, 15, 18, 20]

    const canvasElement = document.getElementById("fluidCanvas") as HTMLCanvasElement;
    // シミュレーション，カメラの初期化
    const mlsmpmFov = 45 * Math.PI / 180  // Lower FOV for more isometric view
    const mlsmpmRadius = params.particleSize
    const mlsmpmDiameter = 2 * mlsmpmRadius
    const mlsmpmZoomRate = 20.0  // Increased zoom rate for distant camera
    const fixedPointMultiplier = 1e7
    const depthMapTexture = device.createTexture({
        label: 'depth map texture', 
        size: [canvas.width, canvas.height, 1],
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        format: 'r32float',
    });
    const depthMapTextureView = depthMapTexture.createView()
    const mlsmpmSimulator = new MLSMPMSimulator(particleBuffer, posvelBuffer, mlsmpmDiameter, device, renderUniformBuffer, depthMapTextureView, canvas, maxGridCount, densityGridBuffer, initBoxSizeBuffer, fixedPointMultiplier, shapeParamsBuffer, uniformManager)
    const mlsmpmRenderer = new FluidRenderer(device, canvas, presentationFormat, mlsmpmRadius, mlsmpmFov, posvelBuffer, renderUniformBuffer,  cubemapTextureView, depthMapTextureView, densityGridBuffer, fixedPointMultiplier, initBoxSizeBuffer)

    console.log("simulator initialization done")

    const camera = new Camera(canvasElement, uniformManager)

    // デバイスロストの監視
    let errorLog = document.getElementById('error-reason') as HTMLSpanElement
    errorLog.textContent = ""
    device.lost.then(info => {
        const reason = info.reason ? `reason: ${info.reason}` : 'unknown reason';
        errorLog.textContent = reason;
    });

    let paramsIdx = -1
    let initBoxSize = [0, 0, 0]
    let realBoxSize = [0, 0, 0]

    let sphereRenderFl = false
    let rotateFl = false
    let boxWidthRatio = 1.

    console.log("simulation start")
    let closingSpeed = 0.
    let prevClosingSpeed = 0.
    async function frame() {
        const start = performance.now();

        const form = document.getElementById("number-button") as HTMLFormElement;
        const selectedValue = typeof params.numParticles === 'string' 
            ? numParticlesOptions.indexOf(params.numParticles)
            : PARTICLE_OPTIONS.findIndex(opt => opt.value === params.numParticles);
        if (params.running && Number(selectedValue) != paramsIdx) {
            paramsIdx = Number(selectedValue)
            // Safety check to prevent accessing undefined array elements
            if (paramsIdx >= mlsmpmNumParticleParams.length) {
                console.warn(`Particle count index ${paramsIdx} is out of bounds. Using last available configuration.`);
                paramsIdx = mlsmpmNumParticleParams.length - 1;
            }
            // Start with the base box size, then calculate proper container size for the shape type
            let baseBoxSize = mlsmpmInitBoxSizes[paramsIdx]
            initBoxSize = mlsmpmSimulator.calculateRequiredBoxSize(shapeTypeMap[params.shapeType] || 0, params, baseBoxSize)
            realBoxSize = [...initBoxSize]
            mlsmpmSimulator.changeBoxSize(realBoxSize)
            mlsmpmSimulator.reset(initBoxSize, mlsmpmNumParticleParams[paramsIdx], shapeTypeMap[params.shapeType] || 0, params)
            // Update camera position based on shape type
            updateCameraPosition(shapeTypeMap[params.shapeType] || 0, initBoxSize, params.sphereRadius)
            let slider = document.getElementById("slider") as HTMLInputElement
            slider.value = "100"
        }

        const particle = document.getElementById("particle") as HTMLInputElement
        sphereRenderFl = particle.checked
        if (params.running) {
            const slider = document.getElementById("slider") as HTMLInputElement
            let curBoxWidthRatio = parseInt(slider.value) / 200 + 0.5
            const maxClosingSpeed = 0.007 * params.speed
            closingSpeed = Math.min(maxClosingSpeed, prevClosingSpeed + maxClosingSpeed / 40.)
            let dVal = Math.min(boxWidthRatio - curBoxWidthRatio, closingSpeed)
            boxWidthRatio -= dVal
            if (dVal <= 0.) {
                closingSpeed = 0.
                prevClosingSpeed = 0.
            } else {
                prevClosingSpeed = closingSpeed
            }   
        }


        realBoxSize[2] = initBoxSize[2] * boxWidthRatio
        mlsmpmSimulator.changeBoxSize(realBoxSize)
        
        // Update shape type
        mlsmpmSimulator.changeShapeType(shapeTypeMap[params.shapeType] || 0)
        
        // Flip gravity logic - simple time-based gravity changes
        if (params.flipGravity) {
            const time = performance.now() * 0.001; // Convert to seconds
            // Create a smooth oscillation between different gravity values
            // This simulates flipping gravity up and down
            const baseGravity = -0.40;
            const amplitude = 0.8; // How much gravity can vary
            const frequency = params.flipSpeed * 0.5; // How fast it oscillates
            params.gravity = baseGravity + amplitude * Math.sin(time * frequency);
        }
        
        // Update gravity, particle size, and shape parameters in uniforms
        renderUniformsViews.gravity.set([params.gravity]);
        renderUniformsViews.sphere_size.set([params.particleSize]);
        
        // Shape parameters are now hardcoded in shaders - no need to update uniforms
        
        device.queue.writeBuffer(renderUniformBuffer, 0, renderUniformsValues) 

        const commandEncoder = device.createCommandEncoder()

        let maxDt = 0.4;
        mlsmpmSimulator.execute(commandEncoder, 
            [camera.currentHoverX / canvas.clientWidth, camera.currentHoverY / canvas.clientHeight], 
            camera.calcMouseVelocity(), mouseRadiuses[paramsIdx] || mouseRadiuses[mouseRadiuses.length - 1], sphereRenderFl, maxDt * params.speed, params.running) 
        let normalizedDiffuseColor = [params.r / 255, params.g / 255, params.b / 255];
        mlsmpmRenderer.execute(context, commandEncoder, mlsmpmSimulator.numParticles, sphereRenderFl, normalizedDiffuseColor, params.colorDensity)

        device.queue.submit([commandEncoder.finish()])

        camera.setNewPrevMouseCoord();
        if (rotateFl) {
            camera.stepAngle();
        }

        const end = performance.now();

        requestAnimationFrame(frame)
    } 
    requestAnimationFrame(frame)
}

main()

