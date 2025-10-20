import { SIMULATION, BUFFER_SIZES } from '../utils/Constants';

export interface BufferConfig {
    label: string;
    size: number;
    usage: GPUBufferUsageFlags;
}

export class BufferManager {
    private device: GPUDevice;
    private buffers: Map<string, GPUBuffer> = new Map();

    constructor(device: GPUDevice) {
        this.device = device;
    }

    createBuffer(key: string, config: BufferConfig): GPUBuffer {
        const buffer = this.device.createBuffer({
            label: config.label,
            size: config.size,
            usage: config.usage,
        });
        this.buffers.set(key, buffer);
        return buffer;
    }

    getBuffer(key: string): GPUBuffer | undefined {
        return this.buffers.get(key);
    }

    writeBuffer(key: string, data: ArrayBuffer, offset: number = 0): void {
        const buffer = this.buffers.get(key);
        if (buffer) {
            this.device.queue.writeBuffer(buffer, offset, data);
        }
    }

    createUniformBuffer(key: string, label: string, size: number): GPUBuffer {
        return this.createBuffer(key, {
            label,
            size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    createStorageBuffer(key: string, label: string, size: number): GPUBuffer {
        return this.createBuffer(key, {
            label,
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }

    // Predefined buffer creators for common use cases
    createParticleBuffer(): GPUBuffer {
        return this.createStorageBuffer(
            'particles',
            'particles buffer',
            SIMULATION.PARTICLE_STRUCT_SIZE * SIMULATION.MAX_PARTICLES
        );
    }

    createCellBuffer(maxGridCount: number): GPUBuffer {
        return this.createStorageBuffer(
            'cells',
            'cells buffer',
            SIMULATION.CELL_STRUCT_SIZE * maxGridCount
        );
    }

    createRenderUniformBuffer(): GPUBuffer {
        return this.createUniformBuffer(
            'renderUniforms',
            'render uniform buffer',
            BUFFER_SIZES.RENDER_UNIFORMS
        );
    }

    createPhysicsUniformBuffer(): GPUBuffer {
        return this.createUniformBuffer(
            'physicsUniforms',
            'physics uniform buffer',
            BUFFER_SIZES.PHYSICS_UNIFORMS
        );
    }

    createMouseInfoBuffer(): GPUBuffer {
        return this.createUniformBuffer(
            'mouseInfo',
            'mouse info buffer',
            BUFFER_SIZES.MOUSE_INFO
        );
    }

    createDensityBuffer(): GPUBuffer {
        return this.createStorageBuffer(
            'density',
            'density buffer',
            4 * SIMULATION.MAX_PARTICLES
        );
    }

    createDtBuffer(): GPUBuffer {
        return this.createUniformBuffer(
            'dt',
            'dt buffer',
            BUFFER_SIZES.DT_BUFFER
        );
    }

    createShapeTypeBuffer(): GPUBuffer {
        return this.createUniformBuffer(
            'shapeType',
            'shape type buffer',
            BUFFER_SIZES.SHAPE_TYPE
        );
    }

    createNumParticlesBuffer(): GPUBuffer {
        return this.createUniformBuffer(
            'numParticles',
            'num particles buffer',
            BUFFER_SIZES.NUM_PARTICLES
        );
    }

    // Cleanup method
    destroy(): void {
        this.buffers.forEach(buffer => buffer.destroy());
        this.buffers.clear();
    }
}

