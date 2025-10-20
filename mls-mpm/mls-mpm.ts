import clearGrid from './clearGrid.wgsl'
import p2g_1 from './p2g_1.wgsl'
import p2g_2 from './p2g_2.wgsl'
import updateGrid from './updateGrid.wgsl'
import g2p from './g2p.wgsl'
import copyPosition from './copyPosition.wgsl'
import p2gDensity from './p2gDensity.wgsl'
import clearDensityGrid from './clearDensityGrid.wgsl'

import { numParticlesMax, renderUniformsViews, physicsUniformsValues, physicsUniformsViews } from '../common'
import { BUFFER_SIZES } from '../utils/Constants'
import { UniformManager } from '../buffers/UniformManager'

export const mlsmpmParticleStructSize = 80

export class MLSMPMSimulator {
    cellStructSize = 16;
    uniformManager: UniformManager
    realBoxSizeBuffer: GPUBuffer
    numParticlesBuffer: GPUBuffer
    densityBuffer: GPUBuffer
    mouseInfoUniformBuffer: GPUBuffer
    sphereRadiusBuffer: GPUBuffer
    initBoxSizeBuffer: GPUBuffer
    shapeTypeBuffer: GPUBuffer
    physicsPropertiesBuffer: GPUBuffer
    shapeParamsBuffer: GPUBuffer
    numParticles = 0
    gridCount = 0
    maxGridCount = 0
    densityGridCount = 0

    clearGridPipeline: GPUComputePipeline
    clearDensityGridPipeline: GPUComputePipeline
    p2g1Pipeline: GPUComputePipeline
    p2g2Pipeline: GPUComputePipeline
    p2gDensityPipeline: GPUComputePipeline
    updateGridPipeline: GPUComputePipeline
    g2pPipeline: GPUComputePipeline
    copyPositionPipeline: GPUComputePipeline

    clearGridBindGroup: GPUBindGroup
    clearDensityGridBindGroup: GPUBindGroup
    p2g1BindGroup: GPUBindGroup
    p2g2BindGroup: GPUBindGroup
    p2gDensityBindGroup: GPUBindGroup
    updateGridBindGroup: GPUBindGroup
    g2pBindGroup: GPUBindGroup
    copyPositionBindGroup: GPUBindGroup

    particleBuffer: GPUBuffer
    dtBuffer: GPUBuffer

    device: GPUDevice

    renderDiameter: number

    frameCount: number

    spawned: boolean

    mouseInfoValues: ArrayBuffer

    restDensity: number

    // Performance monitoring
    private performanceStats = {
        lastFrameTime: 0,
        averageFrameTime: 0,
        frameCount: 0
    };

    // Helper function for optimal workgroup dispatch
    private getWorkgroupCount(count: number, workgroupSize: number = 64): number {
        return Math.ceil(count / workgroupSize);
    }

    // Memory pool for temporary buffers to reduce allocation overhead
    private tempBufferPool: Map<number, ArrayBuffer> = new Map();
    
    private getTempBuffer(size: number): ArrayBuffer {
        if (this.tempBufferPool.has(size)) {
            return this.tempBufferPool.get(size)!;
        }
        const buffer = new ArrayBuffer(size);
        this.tempBufferPool.set(size, buffer);
        return buffer;
    }

    // Performance tracking
    updatePerformanceStats(frameTime: number) {
        this.performanceStats.lastFrameTime = frameTime;
        this.performanceStats.frameCount++;
        
        // Simple moving average
        const alpha = 0.1;
        this.performanceStats.averageFrameTime = 
            alpha * frameTime + (1 - alpha) * this.performanceStats.averageFrameTime;
    }

    getPerformanceStats() {
        return {
            ...this.performanceStats,
            fps: 1000 / this.performanceStats.averageFrameTime
        };
    }

    constructor (particleBuffer: GPUBuffer, posvelBuffer: GPUBuffer, renderDiameter: number, device: GPUDevice, 
        renderUniformBuffer: GPUBuffer, depthMapTextureView: GPUTextureView, canvas: HTMLCanvasElement, 
        maxGridCount: number, densityGridBuffer: GPUBuffer, initBoxSizeBuffer: GPUBuffer, fixedPointMultiplier: number,
        shapeParamsBuffer: GPUBuffer, uniformManager: UniformManager) 
    {
        this.device = device
        this.uniformManager = uniformManager
        this.renderDiameter = renderDiameter
        this.frameCount = 0
        this.spawned = false
        this.numParticles = 0
        this.maxGridCount = maxGridCount
        this.initBoxSizeBuffer = initBoxSizeBuffer
        this.shapeParamsBuffer = shapeParamsBuffer

        const clearGridModule = device.createShaderModule({ code: clearGrid })
        const clearDensityGridModule = device.createShaderModule({ code: clearDensityGrid })
        const p2g1Module = device.createShaderModule({ code: p2g_1 })
        const p2g2Module = device.createShaderModule({ code: p2g_2 })
        const p2gDensityModule = device.createShaderModule({ code: p2gDensity })
        const updateGridModule = device.createShaderModule({ code: updateGrid })
        const g2pModule = device.createShaderModule({ code: g2p })
        const copyPositionModule = device.createShaderModule({ code: copyPosition })

        this.restDensity = 3.

        const constants = {
            stiffness: 50., 
            restDensity: this.restDensity, 
            dynamicViscosity: 0.1, 
            fixedPointMultiplier: fixedPointMultiplier, 
        }

        this.clearGridPipeline = device.createComputePipeline({
            label: "clear grid pipeline", 
            layout: 'auto', 
            compute: {
                module: clearGridModule, 
            }
        })
        this.clearDensityGridPipeline = device.createComputePipeline({
            label: "clear density grid pipeline", 
            layout: 'auto', 
            compute: {
                module: clearDensityGridModule, 
            }
        })
        this.p2g1Pipeline = device.createComputePipeline({
            label: "p2g 1 pipeline", 
            layout: 'auto', 
            compute: {
                module: p2g1Module, 
                constants: {
                    'fixedPointMultiplier': constants.fixedPointMultiplier
                }, 
            }
        })
        this.p2g2Pipeline = device.createComputePipeline({
            label: "p2g 2 pipeline", 
            layout: 'auto', 
            compute: {
                module: p2g2Module, 
                constants: {
                    'fixedPointMultiplier': constants.fixedPointMultiplier, 
                    'stiffness': constants.stiffness, 
                    'restDensity': constants.restDensity, 
                    'dynamicViscosity': constants.dynamicViscosity, 
                }, 
            }
        })
        this.p2gDensityPipeline = device.createComputePipeline({
            label: "p2g density pipeline", 
            layout: 'auto', 
            compute: {
                module: p2gDensityModule, 
                constants: {
                    'densityFixedPointMultiplier': constants.fixedPointMultiplier, 
                }, 
            }
        })
        this.updateGridPipeline = device.createComputePipeline({
            label: "update grid pipeline", 
            layout: 'auto', 
            compute: {
                module: updateGridModule, 
                constants: {
                    'fixedPointMultiplier': constants.fixedPointMultiplier, 
                }, 
            }
        });
        this.g2pPipeline = device.createComputePipeline({
            label: "g2p pipeline", 
            layout: 'auto', 
            compute: {
                module: g2pModule, 
                constants: {
                    'fixedPointMultiplier': constants.fixedPointMultiplier, 
                }, 
            }
        });
        this.copyPositionPipeline = device.createComputePipeline({
            label: "copy position pipeline", 
            layout: 'auto', 
            compute: {
                module: copyPositionModule, 
            }
        });

        const numParticlesValues = new ArrayBuffer(4);
        this.mouseInfoValues = new ArrayBuffer(32);

        const cellBuffer = device.createBuffer({ 
            label: 'cells buffer', 
            size: this.cellStructSize * maxGridCount,  
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        this.densityBuffer = device.createBuffer({
            label: 'density buffer', 
            size: 4 * numParticlesMax, 
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        this.realBoxSizeBuffer = device.createBuffer({
            label: 'real box size buffer', 
            size: 12, // 3 x f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.numParticlesBuffer = device.createBuffer({
            label: 'number of particles buffer', 
            size: numParticlesValues.byteLength, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }) 
        this.mouseInfoUniformBuffer = device.createBuffer({
            label: 'mouse info buffer', 
            size: this.mouseInfoValues.byteLength, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.sphereRadiusBuffer = device.createBuffer({
            label: 'sphere radius buffer', 
            size: 4, // single f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.dtBuffer = device.createBuffer({
            label: 'dt buffer', 
            size: 4, // single f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.shapeTypeBuffer = device.createBuffer({
            label: 'shape type buffer', 
            size: 4, // single u32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.physicsPropertiesBuffer = device.createBuffer({
            label: 'physics properties buffer', 
            size: 16, // 4 f32 values
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        // TODO : これを一か所にまとめる
        const mouseInfoViews = {
            screenSize: new Float32Array(this.mouseInfoValues, 0, 2),
            mouseCoord: new Float32Array(this.mouseInfoValues, 8, 2),
            mouseVel: new Float32Array(this.mouseInfoValues, 16, 2),
            mouseRadius: new Float32Array(this.mouseInfoValues, 24, 1),
        };
        mouseInfoViews.screenSize.set([canvas.width, canvas.height]);
        this.device.queue.writeBuffer(this.mouseInfoUniformBuffer, 0, this.mouseInfoValues);
        
        // Write initial physics properties
        this.device.queue.writeBuffer(this.physicsPropertiesBuffer, 0, physicsUniformsValues);

        // BindGroup
        this.clearGridBindGroup = device.createBindGroup({
            layout: this.clearGridPipeline.getBindGroupLayout(0), 
            entries: [
              { binding: 0, resource: { buffer: cellBuffer }}, 
            ],  
        })
        this.clearDensityGridBindGroup = device.createBindGroup({
            layout: this.clearDensityGridPipeline.getBindGroupLayout(0), 
            entries: [
              { binding: 0, resource: { buffer: densityGridBuffer }}, 
            ],  
        })
        this.p2g1BindGroup = device.createBindGroup({
            layout: this.p2g1Pipeline.getBindGroupLayout(0), 
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }}, 
                { binding: 1, resource: { buffer: cellBuffer }}, 
                { binding: 2, resource: { buffer: initBoxSizeBuffer }}, 
                { binding: 3, resource: { buffer: this.numParticlesBuffer }}, 
            ],  
        })
        this.p2g2BindGroup = device.createBindGroup({
            layout: this.p2g2Pipeline.getBindGroupLayout(0), 
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }}, 
                { binding: 1, resource: { buffer: cellBuffer }}, 
                { binding: 2, resource: { buffer: initBoxSizeBuffer }}, 
                { binding: 3, resource: { buffer: this.numParticlesBuffer }}, 
                { binding: 4, resource: { buffer: this.densityBuffer }}, 
                { binding: 5, resource: { buffer: this.dtBuffer }}, 
            ]
        })
        this.p2gDensityBindGroup = device.createBindGroup({
            layout: this.p2gDensityPipeline.getBindGroupLayout(0), 
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }}, 
                { binding: 1, resource: { buffer: this.densityBuffer }}, 
                { binding: 2, resource: { buffer: this.numParticlesBuffer }}, 
                { binding: 3, resource: { buffer: densityGridBuffer }}, 
                { binding: 4, resource: { buffer: initBoxSizeBuffer }}
            ]
        })
        this.updateGridBindGroup = device.createBindGroup({
            layout: this.updateGridPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: cellBuffer }},
                { binding: 1, resource: { buffer: this.realBoxSizeBuffer }},
                { binding: 2, resource: { buffer: initBoxSizeBuffer }},
                { binding: 3, resource: { buffer: renderUniformBuffer }},
                { binding: 4, resource: depthMapTextureView }, 
                { binding: 5, resource: { buffer: this.mouseInfoUniformBuffer }}, 
                { binding: 6, resource: { buffer: this.dtBuffer }}, 
                { binding: 7, resource: { buffer: this.shapeTypeBuffer }}, 
                { binding: 8, resource: { buffer: this.shapeParamsBuffer }}, 
            ],
        })
        this.g2pBindGroup = device.createBindGroup({
            layout: this.g2pPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }},
                { binding: 1, resource: { buffer: cellBuffer }},
                { binding: 2, resource: { buffer: this.realBoxSizeBuffer }},
                { binding: 3, resource: { buffer: initBoxSizeBuffer }},
                { binding: 4, resource: { buffer: this.numParticlesBuffer }}, 
                { binding: 5, resource: { buffer: this.dtBuffer }}, 
                { binding: 6, resource: { buffer: this.shapeTypeBuffer }}, 
                { binding: 7, resource: { buffer: this.physicsPropertiesBuffer }}, 
                { binding: 8, resource: { buffer: this.shapeParamsBuffer }}, 
            ],
        })
        this.copyPositionBindGroup = device.createBindGroup({
            layout: this.copyPositionPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }}, 
                { binding: 1, resource: { buffer: posvelBuffer }}, 
                { binding: 2, resource: { buffer: this.numParticlesBuffer }}, 
            ]
        })

        this.particleBuffer = particleBuffer
    }

    initDambreak(initBoxSize: number[], numParticles: number, shapeType: number = 0, shapeParams?: any) {
        let particlesBuf = new ArrayBuffer(mlsmpmParticleStructSize * numParticlesMax);
        const spacing = 0.7 ;

        this.numParticles = 0;

        // Provide default shape parameters if not provided
        const defaultShapeParams = {
            boxWidth: initBoxSize[0] - 6,
            boxHeight: initBoxSize[1] * 0.8,
            boxDepth: initBoxSize[2] - 6,
            cylinderRadius: Math.min(initBoxSize[0], initBoxSize[2]) * 0.4,
            cylinderHeight: initBoxSize[1] * 0.8,
            sphereRadius: Math.min(initBoxSize[0], initBoxSize[1], initBoxSize[2]) * 0.4,
            coneRadius: Math.min(initBoxSize[0], initBoxSize[2]) * 0.4,
            coneHeight: initBoxSize[1] * 0.8,
            coneTaper: 0.5
        };
        
        const safeShapeParams = shapeParams || defaultShapeParams;
        let center = [initBoxSize[0] / 2, initBoxSize[1] / 2, initBoxSize[2] / 2];
        
        // Generate particles based on shape type
        switch (shapeType) {
            case 0: // Box
                const boxWidth = shapeParams?.boxWidth ? 
                    shapeParams.boxWidth :  // Use full width without clipping
                    initBoxSize[0] - 6;
                const boxHeight = shapeParams?.boxHeight ? 
                    shapeParams.boxHeight :  // Use full height without clipping
                    initBoxSize[1] * 0.8;
                const boxDepth = shapeParams?.boxDepth ? 
                    shapeParams.boxDepth :  // Use full depth without clipping
                    initBoxSize[2] - 6;
                
                const boxStartX = (initBoxSize[0] - boxWidth) / 2;
                const boxStartZ = (initBoxSize[2] - boxDepth) / 2;
                
                for (let j = 3; j < boxHeight && this.numParticles < numParticles; j += spacing) {
                    for (let i = boxStartX; i < boxStartX + boxWidth && this.numParticles < numParticles; i += spacing) {
                        for (let k = boxStartZ; k < boxStartZ + boxDepth && this.numParticles < numParticles; k += spacing) {
                            this.addParticle(particlesBuf, [i, j, k], spacing);
                        }
                    }
                }
                break;
                
            case 1: // Cylinder
                const cylinderRadius = shapeParams?.cylinderRadius ? 
                    shapeParams.cylinderRadius :  // Use full radius without clipping
                    Math.min(initBoxSize[0], initBoxSize[2]) * 0.4;
                const cylinderHeight = shapeParams?.cylinderHeight ? 
                    shapeParams.cylinderHeight :  // Use full height without clipping
                    initBoxSize[1] * 0.8;
                for (let j = 3; j < cylinderHeight && this.numParticles < numParticles; j += spacing) {
                    for (let r = 0; r < cylinderRadius && this.numParticles < numParticles; r += spacing) {
                        const circumference = 2 * Math.PI * r;
                        const numAngles = Math.max(1, Math.floor(circumference / spacing));
                        for (let a = 0; a < numAngles && this.numParticles < numParticles; a++) {
                            const angle = (a / numAngles) * 2 * Math.PI;
                            const x = center[0] + r * Math.cos(angle);
                            const z = center[2] + r * Math.sin(angle);
                            if (x >= 3 && x < initBoxSize[0] - 3 && z >= 3 && z < initBoxSize[2] - 3) {
                                this.addParticle(particlesBuf, [x, j, z], spacing);
                            }
                        }
                    }
                }
                break;
                
            case 2: // Sphere
                const sphereRadius = shapeParams?.sphereRadius ? 
                    shapeParams.sphereRadius :  // Use full radius without clipping
                    Math.min(initBoxSize[0], initBoxSize[1], initBoxSize[2]) * 0.4;
                
                // Use the exact number of particles requested - let the sphere compress to fit
                const maxParticlesForSphere = Math.min(numParticles, numParticlesMax);
                
                console.log(`Sphere radius: ${sphereRadius}, Requested particles: ${numParticles}, Using: ${maxParticlesForSphere}`);
                
                // Generate particles throughout the entire sphere volume
                // Use a density-based approach that can generate the exact number of particles requested
                const halfRadius = sphereRadius;
                
                // Calculate how many particles we can fit in the sphere with current spacing
                const sphereVolume = (4/3) * Math.PI * sphereRadius * sphereRadius * sphereRadius;
                const particleVolume = spacing * spacing * spacing;
                const maxParticlesInSphere = Math.floor(sphereVolume / particleVolume);
                
                // If we need more particles than can fit with current spacing, use random distribution
                if (maxParticlesForSphere > maxParticlesInSphere) {
                    console.log(`Using random distribution: ${maxParticlesInSphere} particles can fit with spacing ${spacing}, but ${maxParticlesForSphere} requested`);
                    
                    // Generate particles randomly within the sphere
                    for (let i = 0; i < maxParticlesForSphere; i++) {
                        // Generate random point in sphere using rejection sampling
                        let x, y, z, distanceFromCenter;
                        do {
                            x = center[0] + (Math.random() - 0.5) * 2 * sphereRadius;
                            y = center[1] + (Math.random() - 0.5) * 2 * sphereRadius;
                            z = center[2] + (Math.random() - 0.5) * 2 * sphereRadius;
                            distanceFromCenter = Math.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2);
                        } while (distanceFromCenter > sphereRadius);
                        
                        // Check bounds
                        if (x >= 3 && x < initBoxSize[0] - 3 && 
                            y >= 3 && y < initBoxSize[1] - 3 && 
                            z >= 3 && z < initBoxSize[2] - 3) {
                            this.addParticle(particlesBuf, [x, y, z], spacing);
                        }
                    }
                } else {
                    // Use grid-based approach for smaller particle counts
                    for (let x = center[0] - halfRadius; x <= center[0] + halfRadius && this.numParticles < maxParticlesForSphere; x += spacing) {
                        for (let y = center[1] - halfRadius; y <= center[1] + halfRadius && this.numParticles < maxParticlesForSphere; y += spacing) {
                            for (let z = center[2] - halfRadius; z <= center[2] + halfRadius && this.numParticles < maxParticlesForSphere; z += spacing) {
                                // Check if point is inside the sphere
                                const dx = x - center[0];
                                const dy = y - center[1];
                                const dz = z - center[2];
                                const distanceFromCenter = Math.sqrt(dx * dx + dy * dy + dz * dz);
                                
                                // Add particles anywhere inside the sphere (including surface)
                                if (distanceFromCenter <= sphereRadius) {
                                    // Check bounds
                                    if (x >= 3 && x < initBoxSize[0] - 3 && 
                                        y >= 3 && y < initBoxSize[1] - 3 && 
                                        z >= 3 && z < initBoxSize[2] - 3) {
                                        this.addParticle(particlesBuf, [x, y, z], spacing);
                                    }
                                }
                            }
                        }
                    }
                }
                break;
                
            case 3: // Cone
                const coneRadius = shapeParams?.coneRadius ? 
                    shapeParams.coneRadius :  // Use full radius without clipping
                    Math.min(initBoxSize[0], initBoxSize[2]) * 0.4;
                const coneHeight = shapeParams?.coneHeight ? 
                    shapeParams.coneHeight :  // Use full height without clipping
                    initBoxSize[1] * 0.8;
                const coneTaper = shapeParams?.coneTaper || 0.5;
                for (let j = 3; j < coneHeight && this.numParticles < numParticles; j += spacing) {
                    const heightFactor = j / coneHeight;
                    const currentRadius = coneRadius * (1 - heightFactor * coneTaper);
                    for (let r = 0; r < currentRadius && this.numParticles < numParticles; r += spacing) {
                        const circumference = 2 * Math.PI * r;
                        const numAngles = Math.max(1, Math.floor(circumference / spacing));
                        for (let a = 0; a < numAngles && this.numParticles < numParticles; a++) {
                            const angle = (a / numAngles) * 2 * Math.PI;
                            const x = center[0] + r * Math.cos(angle);
                            const z = center[2] + r * Math.sin(angle);
                            if (x >= 3 && x < initBoxSize[0] - 3 && z >= 3 && z < initBoxSize[2] - 3) {
                                this.addParticle(particlesBuf, [x, j, z], spacing);
                            }
                        }
                    }
                }
                break;
                
            case 4: // Triple Helix Venturi
                const venturiTopRadius = shapeParams?.venturiTopRadius ? 
                    shapeParams.venturiTopRadius : 
                    Math.min(initBoxSize[0], initBoxSize[2]) * 0.4;
                const venturiThroatRadius = shapeParams?.venturiThroatRadius ? 
                    shapeParams.venturiThroatRadius : 
                    Math.min(initBoxSize[0], initBoxSize[2]) * 0.12;
                const venturiBottomRadius = shapeParams?.venturiBottomRadius ? 
                    shapeParams.venturiBottomRadius : 
                    Math.min(initBoxSize[0], initBoxSize[2]) * 0.5;
                const venturiHeight = shapeParams?.venturiHeight ? 
                    shapeParams.venturiHeight : 
                    initBoxSize[1] * 0.9;
                const venturiThroatPosition = shapeParams?.venturiThroatPosition || 0.5;
                const helixCount = shapeParams?.venturiHelixCount || 3;
                const helixPitch = shapeParams?.venturiHelixPitch || 0.3;
                
                // Generate particles efficiently for helix Venturi - use direct helix generation
                const inletHeight = venturiHeight * 0.8; // Fill most of the container
                const particlesPerHelix = Math.floor(numParticles / helixCount); // Distribute particles across helices
                
                for (let helix = 0; helix < helixCount && this.numParticles < numParticles; helix++) {
                    const helixAngle = (helix / helixCount) * 2 * Math.PI;
                    let helixParticleCount = 0;
                    
                    for (let j = 3; j < inletHeight && helixParticleCount < particlesPerHelix && this.numParticles < numParticles; j += spacing) {
                        // Calculate radius at current height using smooth interpolation
                        const t = j / venturiHeight;
                        let currentRadius;
                        if (t < venturiThroatPosition) {
                            const localT = t / venturiThroatPosition;
                            const eased = 1 - Math.pow(1 - localT, 3); // ease-out cubic
                            currentRadius = venturiTopRadius + (venturiThroatRadius - venturiTopRadius) * eased;
                        } else {
                            const localT = (t - venturiThroatPosition) / (1 - venturiThroatPosition);
                            const eased = Math.pow(localT, 3); // ease-in cubic
                            currentRadius = venturiThroatRadius + (venturiBottomRadius - venturiThroatRadius) * eased;
                        }
                        
                        // Calculate helix position at this height
                        const helixOffset = t * helixPitch * 2 * Math.PI;
                        const currentHelixAngle = helixAngle + helixOffset;
                        
                        const helixCenterX = center[0] + currentRadius * 0.6 * Math.cos(currentHelixAngle);
                        const helixCenterZ = center[2] + currentRadius * 0.6 * Math.sin(currentHelixAngle);
                        const helixRadius = currentRadius * 0.4;
                        
                        // Generate particles in a circular pattern around the helix center
                        const maxRadius = Math.floor(helixRadius / spacing);
                        for (let r = 0; r < maxRadius && helixParticleCount < particlesPerHelix && this.numParticles < numParticles; r += 1) {
                            const circumference = 2 * Math.PI * r;
                            const numAngles = Math.max(1, Math.floor(circumference / spacing));
                            for (let a = 0; a < numAngles && helixParticleCount < particlesPerHelix && this.numParticles < numParticles; a++) {
                                const angle = (a / numAngles) * 2 * Math.PI;
                                const x = helixCenterX + r * spacing * Math.cos(angle);
                                const z = helixCenterZ + r * spacing * Math.sin(angle);
                                
                                // Add particle if within bounds
                                if (x >= 3 && x < initBoxSize[0] - 3 && 
                                    z >= 3 && z < initBoxSize[2] - 3) {
                                    this.addParticle(particlesBuf, [x, j, z], spacing);
                                    helixParticleCount++;
                                }
                            }
                        }
                    }
                }
                
                // If we still have room for more particles, fill any remaining space
                if (this.numParticles < numParticles) {
                    const remainingParticles = numParticles - this.numParticles;
                    const particlesPerHeight = Math.floor(remainingParticles / inletHeight);
                    
                    for (let j = 3; j < inletHeight && this.numParticles < numParticles; j += spacing) {
                        const t = j / venturiHeight;
                        let currentRadius;
                        if (t < venturiThroatPosition) {
                            const localT = t / venturiThroatPosition;
                            const eased = 1 - Math.pow(1 - localT, 3);
                            currentRadius = venturiTopRadius + (venturiThroatRadius - venturiTopRadius) * eased;
                        } else {
                            const localT = (t - venturiThroatPosition) / (1 - venturiThroatPosition);
                            const eased = Math.pow(localT, 3);
                            currentRadius = venturiThroatRadius + (venturiBottomRadius - venturiThroatRadius) * eased;
                        }
                        
                        // Fill remaining space with random particles in helix strands
                        for (let i = 0; i < particlesPerHeight && this.numParticles < numParticles; i++) {
                            const helix = Math.floor(Math.random() * helixCount);
                            const helixAngle = (helix / helixCount) * 2 * Math.PI;
                            const helixOffset = t * helixPitch * 2 * Math.PI;
                            const currentHelixAngle = helixAngle + helixOffset;
                            
                            const helixCenterX = center[0] + currentRadius * 0.6 * Math.cos(currentHelixAngle);
                            const helixCenterZ = center[2] + currentRadius * 0.6 * Math.sin(currentHelixAngle);
                            const helixRadius = currentRadius * 0.4;
                            
                            // Random position within helix
                            const r = Math.random() * helixRadius;
                            const angle = Math.random() * 2 * Math.PI;
                            const x = helixCenterX + r * Math.cos(angle);
                            const z = helixCenterZ + r * Math.sin(angle);
                            
                            if (x >= 3 && x < initBoxSize[0] - 3 && 
                                z >= 3 && z < initBoxSize[2] - 3) {
                                this.addParticle(particlesBuf, [x, j, z], spacing);
                            }
                        }
                    }
                }
                break;
                
            default: // Default to box
                for (let j = 3; j < initBoxSize[1] * 0.80 && this.numParticles < numParticles; j += spacing) {
                    for (let i = initBoxSize[0] * 0.25; i < initBoxSize[0] - 4 && this.numParticles < numParticles; i += spacing) {
                        for (let k = 3; k < initBoxSize[2] * 0.95 && this.numParticles < numParticles; k += spacing) {
                            this.addParticle(particlesBuf, [i, j, k], spacing);
                        }
                    }
                }
                break;
        }
        
        console.log(this.numParticles)
        
        let particles = new ArrayBuffer(mlsmpmParticleStructSize * this.numParticles);
        const oldView = new Uint8Array(particlesBuf);
        const newView = new Uint8Array(particles);
        newView.set(oldView.subarray(0, newView.length));
        
        return particles;
    }

    private addParticle(particlesBuf: ArrayBuffer, position: number[], spacing: number) {
        const offset = mlsmpmParticleStructSize * this.numParticles;
        const particleViews = {
            position: new Float32Array(particlesBuf, offset + 0, 3),
            v: new Float32Array(particlesBuf, offset + 16, 3),
            C: new Float32Array(particlesBuf, offset + 32, 12),
        };
        const jitter = 0.5 * Math.random();
        particleViews.position.set([
            position[0] + jitter, 
            position[1] + jitter, 
            position[2] + jitter
        ]);
        
        // Initialize velocity to zero for particles at rest
        particleViews.v.set([0, 0, 0]);
        
        // Initialize deformation gradient to identity matrix
        particleViews.C.set([
            1, 0, 0, 0,  // First row of 3x3 identity matrix
            0, 1, 0, 0,  // Second row
            0, 0, 1, 0   // Third row
        ]);
        
        this.numParticles++;
    }

    reset(initBoxSize: number[], numParticles: number, shapeType: number = 0, shapeParams?: any) {
        renderUniformsViews.sphere_size.set([this.renderDiameter])
        this.gridCount = Math.ceil(initBoxSize[0]) * Math.ceil(initBoxSize[1]) * Math.ceil(initBoxSize[2])
        if (this.gridCount > this.maxGridCount) {
            throw new Error("gridCount should be equal to or less than maxGridCount")
        }
        this.densityGridCount = this.gridCount 
        const initBoxSizeValues = new ArrayBuffer(12)
        const initBoxSizeViews = new Float32Array(initBoxSizeValues)
        initBoxSizeViews.set(initBoxSize);    
        this.device.queue.writeBuffer(this.initBoxSizeBuffer, 0, initBoxSizeValues)
        this.frameCount = 0;
        let particles = this.initDambreak(initBoxSize, numParticles, shapeType, shapeParams)
        this.device.queue.writeBuffer(this.particleBuffer, 0, particles)
        this.changeBoxSize(initBoxSize)
        this.changeNumParticles(this.numParticles)
    }

    execute(commandEncoder: GPUCommandEncoder, mouseCoord: number[], mouseVel: number[], mouseRadius: number, 
        densityGridFlag: boolean, dt: number, running: boolean) {
        const computePass = commandEncoder.beginComputePass();

        const canvasInfoViews = {
            screenSize: new Float32Array(this.mouseInfoValues, 0, 2),
            mouseCoord: new Float32Array(this.mouseInfoValues, 8, 2),
            mouseVel: new Float32Array(this.mouseInfoValues, 16, 2),
            mouseRadius: new Float32Array(this.mouseInfoValues, 24, 1),
        };
        canvasInfoViews.mouseCoord.set([mouseCoord[0], mouseCoord[1]])
        canvasInfoViews.mouseVel.set([mouseVel[0], mouseVel[1]])
        canvasInfoViews.mouseRadius.set([mouseRadius])
        this.device.queue.writeBuffer(this.mouseInfoUniformBuffer, 0, this.mouseInfoValues);

        // Use memory pool for temporary buffers
        const dtValues = this.getTempBuffer(4)
        const dtViews = new Float32Array(dtValues)
        dtViews.set([dt])
        this.device.queue.writeBuffer(this.dtBuffer, 0, dtViews)


        if (!densityGridFlag) { // 通常
            if (running) {
                for (let i = 0; i < 1; i++) {  // single timestep!!!
                    computePass.setBindGroup(0, this.clearGridBindGroup);
                    computePass.setPipeline(this.clearGridPipeline);
                    computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 64)) 
                    computePass.setBindGroup(0, this.p2g1BindGroup)
                    computePass.setPipeline(this.p2g1Pipeline)
                    computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))
                    computePass.setBindGroup(0, this.p2g2BindGroup)
                    computePass.setPipeline(this.p2g2Pipeline)
                    computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64)) 
                    computePass.setBindGroup(0, this.updateGridBindGroup)
                    computePass.setPipeline(this.updateGridPipeline)
                    computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 64)) 
                    computePass.setBindGroup(0, this.g2pBindGroup)
                    computePass.setPipeline(this.g2pPipeline)
                    computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64)) 
                }
                computePass.setBindGroup(0, this.copyPositionBindGroup)
                computePass.setPipeline(this.copyPositionPipeline)
                computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))  
            }
        } else { // density grid を更新する場合
            console.log("density grid path")
            if (running) {
                for (let i = 0; i < 1; i++) {  // single timestep!!!
                    computePass.setBindGroup(0, this.clearGridBindGroup);
                    computePass.setPipeline(this.clearGridPipeline);
                    computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 64)) 
                    computePass.setBindGroup(0, this.p2g1BindGroup)
                    computePass.setPipeline(this.p2g1Pipeline)
                    computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))
                    computePass.setBindGroup(0, this.p2g2BindGroup)
                    computePass.setPipeline(this.p2g2Pipeline)
                    computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64)) 
                    computePass.setBindGroup(0, this.updateGridBindGroup)
                    computePass.setPipeline(this.updateGridPipeline)
                    computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 64)) 
                    computePass.setBindGroup(0, this.g2pBindGroup)
                    computePass.setPipeline(this.g2pPipeline)
                    computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64)) 
                }
            }
            // density grid をクリア
            computePass.setBindGroup(0, this.clearDensityGridBindGroup)
            computePass.setPipeline(this.clearDensityGridPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.densityGridCount / 64))
            // density grid の p2g
            computePass.setBindGroup(0, this.p2gDensityBindGroup)
            computePass.setPipeline(this.p2gDensityPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))

            computePass.setBindGroup(0, this.copyPositionBindGroup)
            computePass.setPipeline(this.copyPositionPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))  
        }

        computePass.end()

        this.frameCount++;
    }

    changeBoxSize(realBoxSize: number[]) {
        const realBoxSizeValues = new ArrayBuffer(12);
        const realBoxSizeViews = new Float32Array(realBoxSizeValues);
        realBoxSizeViews.set(realBoxSize)
        this.device.queue.writeBuffer(this.realBoxSizeBuffer, 0, realBoxSizeViews)
    }

    changeNumParticles(numParticles: number) {
        const numParticlesValues = new ArrayBuffer(4);
        const numParticlesViews = new Int32Array(numParticlesValues)
        numParticlesViews.set([numParticles])
        this.device.queue.writeBuffer(this.numParticlesBuffer, 0, numParticlesViews)
        this.numParticles = numParticles
    }

    changeShapeType(shapeType: number) {
        const shapeTypeValues = new ArrayBuffer(4);
        const shapeTypeViews = new Uint32Array(shapeTypeValues)
        shapeTypeViews.set([shapeType])
        this.device.queue.writeBuffer(this.shapeTypeBuffer, 0, shapeTypeViews)
    }

    updatePhysicsProperties(viscosity: number, wallStiffness: number, collisionDamping: number, velocityCap: number) {
        // Use UniformManager as single source of truth
        this.uniformManager.updatePhysicsProperties(viscosity, wallStiffness, collisionDamping, velocityCap);
        this.device.queue.writeBuffer(this.physicsPropertiesBuffer, 0, this.uniformManager.physicsUniformsValues);
    }

    updateShapeParameters(shapeParams: any) {
        const shapeParamsValues = new ArrayBuffer(BUFFER_SIZES.SHAPE_PARAMS);
        const shapeParamsViews = new Float32Array(shapeParamsValues);
        shapeParamsViews.set([
            shapeParams.boxWidth,
            shapeParams.boxHeight,
            shapeParams.boxDepth,
            shapeParams.cylinderRadius,
            shapeParams.cylinderHeight,
            shapeParams.sphereRadius,
            shapeParams.coneRadius,
            shapeParams.coneHeight,
            shapeParams.coneTaper,
            shapeParams.venturiTopRadius,
            shapeParams.venturiThroatRadius,
            shapeParams.venturiBottomRadius,
            shapeParams.venturiHeight,
            shapeParams.venturiThroatPosition,
            shapeParams.venturiHelixCount,
            shapeParams.venturiHelixPitch,
            0.0 // padding
        ]);
        this.device.queue.writeBuffer(this.shapeParamsBuffer, 0, shapeParamsValues);
    }

    calculateRequiredBoxSize(shapeType: number, shapeParams: any, currentBoxSize: number[]): number[] {
        switch (shapeType) {
            case 0: // Box
                // The simulation box IS the container box
                return [
                    shapeParams.boxWidth + 6,  // Add small padding for boundaries
                    shapeParams.boxHeight + 6,
                    shapeParams.boxDepth + 6
                ];
            case 1: // Cylinder
                // The simulation box should be sized to fit the cylinder
                const cylinderDiameter = shapeParams.cylinderRadius * 2;
                return [
                    cylinderDiameter + 6,  // Width = diameter + padding
                    shapeParams.cylinderHeight + 6,  // Height = cylinder height + padding
                    cylinderDiameter + 6   // Depth = diameter + padding
                ];
            case 2: // Sphere
                // Calculate container size to fit the sphere with extra space (within grid limits)
                const sphereDiameter = shapeParams.sphereRadius * 2;
                const spherePadding = 60; // More space around the sphere, but within 140x140x140 grid limit
                return [
                    Math.min(sphereDiameter + spherePadding, 140), // Cap at 140 to fit grid
                    Math.min(sphereDiameter + spherePadding, 140),
                    Math.min(sphereDiameter + spherePadding, 140)
                ];
            case 3: // Cone
                // The simulation box should be sized to fit the cone
                const coneDiameter = shapeParams.coneRadius * 2;
                return [
                    coneDiameter + 6,  // Width = base diameter + padding
                    shapeParams.coneHeight + 6,  // Height = cone height + padding
                    coneDiameter + 6   // Depth = base diameter + padding
                ];
            case 4: // Triple Helix Venturi
                // The simulation box should be sized to fit the widest part of the Venturi tube
                const maxVenturiRadius = Math.max(shapeParams.venturiTopRadius, shapeParams.venturiBottomRadius);
                const venturiDiameter = maxVenturiRadius * 2;
                return [
                    venturiDiameter + 6,  // Width = max diameter + padding
                    shapeParams.venturiHeight + 6,  // Height = tube height + padding
                    venturiDiameter + 6   // Depth = max diameter + padding
                ];
            default:
                return currentBoxSize;
        }
    }

}