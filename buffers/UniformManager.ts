import { PHYSICS, RENDERING, BUFFER_SIZES, CONTAINERS } from '../utils/Constants';

export class UniformManager {
    private device: GPUDevice;
    private bufferManager: any; // Will be BufferManager type

    // Uniform data arrays
    public renderUniformsValues: ArrayBuffer;
    public physicsUniformsValues: ArrayBuffer;
    public mouseInfoValues: ArrayBuffer;
    public shapeParamsValues: ArrayBuffer;

    // Typed views for easy access
    public renderUniformsViews: any;
    public physicsUniformsViews: any;
    public mouseInfoViews: any;
    public shapeParamsViews: any;

    constructor(device: GPUDevice, bufferManager: any) {
        this.device = device;
        this.bufferManager = bufferManager;
        
        this.initializeUniforms();
    }

    private initializeUniforms(): void {
        // Render uniforms
        this.renderUniformsValues = new ArrayBuffer(BUFFER_SIZES.RENDER_UNIFORMS);
        this.renderUniformsViews = {
            texel_size: new Float32Array(this.renderUniformsValues, 0, 2),
            sphere_size: new Float32Array(this.renderUniformsValues, 8, 2),
            inv_projection_matrix: new Float32Array(this.renderUniformsValues, 16, 16),
            projection_matrix: new Float32Array(this.renderUniformsValues, 80, 16),
            view_matrix: new Float32Array(this.renderUniformsValues, 144, 16),
            inv_view_matrix: new Float32Array(this.renderUniformsValues, 208, 16),
            gravity: new Float32Array(this.renderUniformsValues, 272, 1),
        };

        // Physics uniforms
        this.physicsUniformsValues = new ArrayBuffer(BUFFER_SIZES.PHYSICS_UNIFORMS);
        this.physicsUniformsViews = {
            viscosity: new Float32Array(this.physicsUniformsValues, 0, 1),
            wallStiffness: new Float32Array(this.physicsUniformsValues, 4, 1),
            collisionDamping: new Float32Array(this.physicsUniformsValues, 8, 1),
            velocityCap: new Float32Array(this.physicsUniformsValues, 12, 1),
        };

        // Mouse info
        this.mouseInfoValues = new ArrayBuffer(BUFFER_SIZES.MOUSE_INFO);
        this.mouseInfoViews = {
            screenSize: new Float32Array(this.mouseInfoValues, 0, 2),
            mouseCoord: new Float32Array(this.mouseInfoValues, 8, 2),
            mouseVel: new Float32Array(this.mouseInfoValues, 16, 2),
            mouseRadius: new Float32Array(this.mouseInfoValues, 24, 1),
        };

        // Shape parameters
        this.shapeParamsValues = new ArrayBuffer(BUFFER_SIZES.SHAPE_PARAMS);
        this.shapeParamsViews = {
            boxWidth: new Float32Array(this.shapeParamsValues, 0, 1),
            boxHeight: new Float32Array(this.shapeParamsValues, 4, 1),
            boxDepth: new Float32Array(this.shapeParamsValues, 8, 1),
            cylinderRadius: new Float32Array(this.shapeParamsValues, 12, 1),
            cylinderHeight: new Float32Array(this.shapeParamsValues, 16, 1),
            sphereRadius: new Float32Array(this.shapeParamsValues, 20, 1),
            coneRadius: new Float32Array(this.shapeParamsValues, 24, 1),
            coneHeight: new Float32Array(this.shapeParamsValues, 28, 1),
            coneTaper: new Float32Array(this.shapeParamsValues, 32, 1),
            // Padding for alignment
            padding: new Float32Array(this.shapeParamsValues, 36, 1),
        };

        this.setDefaultValues();
    }

    private setDefaultValues(): void {
        // Set default render values
        this.renderUniformsViews.texel_size.set(RENDERING.DEFAULT_TEXEL_SIZE);
        this.renderUniformsViews.sphere_size.set(RENDERING.DEFAULT_SPHERE_SIZE);
        this.renderUniformsViews.gravity.set([PHYSICS.DEFAULT_GRAVITY]);

        // Set default physics values
        this.physicsUniformsViews.viscosity.set([PHYSICS.DEFAULT_VISCOSITY]);
        this.physicsUniformsViews.wallStiffness.set([PHYSICS.DEFAULT_WALL_STIFFNESS]);
        this.physicsUniformsViews.collisionDamping.set([PHYSICS.DEFAULT_COLLISION_DAMPING]);
        this.physicsUniformsViews.velocityCap.set([PHYSICS.DEFAULT_VELOCITY_CAP]);

        // Set default shape parameter values
        this.shapeParamsViews.boxWidth.set([CONTAINERS.BOX.DEFAULT_WIDTH]);
        this.shapeParamsViews.boxHeight.set([CONTAINERS.BOX.DEFAULT_HEIGHT]);
        this.shapeParamsViews.boxDepth.set([CONTAINERS.BOX.DEFAULT_DEPTH]);
        this.shapeParamsViews.cylinderRadius.set([CONTAINERS.CYLINDER.DEFAULT_RADIUS]);
        this.shapeParamsViews.cylinderHeight.set([CONTAINERS.CYLINDER.DEFAULT_HEIGHT]);
        this.shapeParamsViews.sphereRadius.set([CONTAINERS.SPHERE.DEFAULT_RADIUS]);
        this.shapeParamsViews.coneRadius.set([CONTAINERS.CONE.DEFAULT_RADIUS]);
        this.shapeParamsViews.coneHeight.set([CONTAINERS.CONE.DEFAULT_HEIGHT]);
        this.shapeParamsViews.coneTaper.set([CONTAINERS.CONE.DEFAULT_TAPER]);
    }

    // Render uniform methods
    updateTexelSize(width: number, height: number): void {
        this.renderUniformsViews.texel_size.set([1.0 / width, 1.0 / height]);
        this.bufferManager.writeBuffer('renderUniforms', this.renderUniformsValues);
    }

    updateGravity(gravity: number): void {
        this.renderUniformsViews.gravity.set([gravity]);
        this.bufferManager.writeBuffer('renderUniforms', this.renderUniformsValues);
    }

    updateMatrices(matrices: {
        invProjection?: Float32Array;
        projection?: Float32Array;
        view?: Float32Array;
        invView?: Float32Array;
    }): void {
        if (matrices.invProjection) {
            this.renderUniformsViews.inv_projection_matrix.set(matrices.invProjection);
        }
        if (matrices.projection) {
            this.renderUniformsViews.projection_matrix.set(matrices.projection);
        }
        if (matrices.view) {
            this.renderUniformsViews.view_matrix.set(matrices.view);
        }
        if (matrices.invView) {
            this.renderUniformsViews.inv_view_matrix.set(matrices.invView);
        }
        this.bufferManager.writeBuffer('renderUniforms', this.renderUniformsValues);
    }

    // Physics uniform methods
    updatePhysicsProperties(viscosity: number, wallStiffness: number, collisionDamping: number, velocityCap: number): void {
        this.physicsUniformsViews.viscosity.set([viscosity]);
        this.physicsUniformsViews.wallStiffness.set([wallStiffness]);
        this.physicsUniformsViews.collisionDamping.set([collisionDamping]);
        this.physicsUniformsViews.velocityCap.set([velocityCap]);
        this.bufferManager.writeBuffer('physicsUniforms', this.physicsUniformsValues);
    }

    // Mouse info methods
    updateMouseInfo(mouseCoord: number[], mouseVel: number[], mouseRadius: number, screenSize?: [number, number]): void {
        this.mouseInfoViews.mouseCoord.set(mouseCoord);
        this.mouseInfoViews.mouseVel.set(mouseVel);
        this.mouseInfoViews.mouseRadius.set([mouseRadius]);
        if (screenSize) {
            this.mouseInfoViews.screenSize.set(screenSize);
        }
        this.bufferManager.writeBuffer('mouseInfo', this.mouseInfoValues);
    }

    // Shape parameters methods
    updateShapeParameters(shapeParams: any): void {
        this.shapeParamsViews.boxWidth.set([shapeParams.boxWidth]);
        this.shapeParamsViews.boxHeight.set([shapeParams.boxHeight]);
        this.shapeParamsViews.boxDepth.set([shapeParams.boxDepth]);
        this.shapeParamsViews.cylinderRadius.set([shapeParams.cylinderRadius]);
        this.shapeParamsViews.cylinderHeight.set([shapeParams.cylinderHeight]);
        this.shapeParamsViews.sphereRadius.set([shapeParams.sphereRadius]);
        this.shapeParamsViews.coneRadius.set([shapeParams.coneRadius]);
        this.shapeParamsViews.coneHeight.set([shapeParams.coneHeight]);
        this.shapeParamsViews.coneTaper.set([shapeParams.coneTaper]);
        this.bufferManager.writeBuffer('shapeParams', this.shapeParamsValues);
    }

    // Utility methods
    writeAllUniforms(): void {
        this.bufferManager.writeBuffer('renderUniforms', this.renderUniformsValues);
        this.bufferManager.writeBuffer('physicsUniforms', this.physicsUniformsValues);
        this.bufferManager.writeBuffer('mouseInfo', this.mouseInfoValues);
        this.bufferManager.writeBuffer('shapeParams', this.shapeParamsValues);
    }
}

