import clearGrid from '../mls-mpm/clearGrid.wgsl';
import p2g_1 from '../mls-mpm/p2g_1.wgsl';
import p2g_2 from '../mls-mpm/p2g_2.wgsl';
import updateGrid from '../mls-mpm/updateGrid.wgsl';
import g2p from '../mls-mpm/g2p.wgsl';
import copyPosition from '../mls-mpm/copyPosition.wgsl';
import p2gDensity from '../mls-mpm/p2gDensity.wgsl';
import clearDensityGrid from '../mls-mpm/clearDensityGrid.wgsl';

export class ShaderManager {
    private device: GPUDevice;
    private modules: Map<string, GPUShaderModule> = new Map();
    private pipelines: Map<string, GPUComputePipeline> = new Map();

    constructor(device: GPUDevice) {
        this.device = device;
        this.compileShaders();
    }

    private compileShaders(): void {
        // Compute shaders
        this.compileShader('clearGrid', clearGrid);
        this.compileShader('p2g_1', p2g_1);
        this.compileShader('p2g_2', p2g_2);
        this.compileShader('updateGrid', updateGrid);
        this.compileShader('g2p', g2p);
        this.compileShader('copyPosition', copyPosition);
        this.compileShader('p2gDensity', p2gDensity);
        this.compileShader('clearDensityGrid', clearDensityGrid);
    }

    private compileShader(name: string, source: string): void {
        try {
            const module = this.device.createShaderModule({
                label: `${name} shader`,
                code: source,
            });
            this.modules.set(name, module);
        } catch (error) {
            console.error(`Failed to compile shader ${name}:`, error);
            throw error;
        }
    }

    getShader(name: string): GPUShaderModule | undefined {
        return this.modules.get(name);
    }

    createComputePipeline(name: string, shaderName: string, bindGroupLayout: GPUBindGroupLayout): GPUComputePipeline {
        const shader = this.getShader(shaderName);
        if (!shader) {
            throw new Error(`Shader ${shaderName} not found`);
        }

        const pipeline = this.device.createComputePipeline({
            label: `${name} pipeline`,
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
            }),
            compute: {
                module: shader,
            },
        });

        this.pipelines.set(name, pipeline);
        return pipeline;
    }

    getPipeline(name: string): GPUComputePipeline | undefined {
        return this.pipelines.get(name);
    }

    // Predefined pipeline creators for common use cases
    createMLSMPMPipelines(bindGroupLayout: GPUBindGroupLayout): {
        clearGrid: GPUComputePipeline;
        p2g1: GPUComputePipeline;
        p2g2: GPUComputePipeline;
        updateGrid: GPUComputePipeline;
        g2p: GPUComputePipeline;
        copyPosition: GPUComputePipeline;
        p2gDensity: GPUComputePipeline;
        clearDensityGrid: GPUComputePipeline;
    } {
        return {
            clearGrid: this.createComputePipeline('clearGrid', 'clearGrid', bindGroupLayout),
            p2g1: this.createComputePipeline('p2g1', 'p2g_1', bindGroupLayout),
            p2g2: this.createComputePipeline('p2g2', 'p2g_2', bindGroupLayout),
            updateGrid: this.createComputePipeline('updateGrid', 'updateGrid', bindGroupLayout),
            g2p: this.createComputePipeline('g2p', 'g2p', bindGroupLayout),
            copyPosition: this.createComputePipeline('copyPosition', 'copyPosition', bindGroupLayout),
            p2gDensity: this.createComputePipeline('p2gDensity', 'p2gDensity', bindGroupLayout),
            clearDensityGrid: this.createComputePipeline('clearDensityGrid', 'clearDensityGrid', bindGroupLayout),
        };
    }

    // Cleanup method
    destroy(): void {
        this.modules.clear();
        this.pipelines.clear();
    }
}

