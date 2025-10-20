import { 
    PHYSICS, 
    RENDERING, 
    CONTAINERS, 
    PARTICLE_OPTIONS, 
    SHAPE_OPTIONS, 
    CAMERA,
    GUI 
} from '../utils/Constants';

export interface SimulationParameters {
    // Physics
    gravity: number;
    particleSize: number;
    flipGravity: boolean;
    flipSpeed: number;
    viscosity: number;
    wallStiffness: number;
    collisionDamping: number;
    velocityCap: number;

    // Rendering
    sigma: number;
    speed: number;
    colorDensity: number;
    r: number;
    g: number;
    b: number;

    // Particles
    numParticles: number | string;

    // Shapes
    shapeType: string;
    boxWidth: number;
    boxHeight: number;
    boxDepth: number;
    cylinderRadius: number;
    cylinderHeight: number;
    sphereRadius: number;
    coneRadius: number;
    coneHeight: number;
    coneTaper: number;
    venturiTopRadius: number;
    venturiThroatRadius: number;
    venturiBottomRadius: number;
    venturiHeight: number;
    venturiThroatPosition: number;
    venturiHelixCount: number;
    venturiHelixPitch: number;
    torusMajorRadius: number;
    torusMinorRadius: number;
    torusHeight: number;
    torusKnotCount: number;
    torusKnotIntensity: number;
    torusTwistAmount: number;
    chemotaxisEnabled: boolean;
    chemotaxisStrength: number;
    chemotaxisDiffusionRate: number;
    chemotaxisDecayRate: number;
    chemotaxisSourceRadius: number;
    chemotaxisAttraction: number;

    // Simulation state
    running: boolean;

    // GUI functions
    toggleSimulation?: () => void;
    toggleFlipGravity?: () => void;
}

export class ParameterManager {
    private parameters: SimulationParameters;

    constructor() {
        this.parameters = this.createDefaultParameters();
    }

    private createDefaultParameters(): SimulationParameters {
        return {
            // Physics
            gravity: PHYSICS.DEFAULT_GRAVITY,
            particleSize: PHYSICS.DEFAULT_PARTICLE_SIZE,
            flipGravity: false,
            flipSpeed: PHYSICS.DEFAULT_FLIP_SPEED,
            viscosity: PHYSICS.DEFAULT_VISCOSITY,
            wallStiffness: PHYSICS.DEFAULT_WALL_STIFFNESS,
            collisionDamping: PHYSICS.DEFAULT_COLLISION_DAMPING,
            velocityCap: PHYSICS.DEFAULT_VELOCITY_CAP,

            // Rendering
            sigma: RENDERING.DEFAULT_SIGMA,
            speed: RENDERING.DEFAULT_SPEED,
            colorDensity: RENDERING.DEFAULT_COLOR_DENSITY,
            r: RENDERING.DEFAULT_COLOR.r,
            g: RENDERING.DEFAULT_COLOR.g,
            b: RENDERING.DEFAULT_COLOR.b,

            // Particles
            numParticles: PARTICLE_OPTIONS[1].value,

            // Shapes
            shapeType: SHAPE_OPTIONS[1], // Cylinder
            boxWidth: CONTAINERS.BOX.DEFAULT_WIDTH,
            boxHeight: CONTAINERS.BOX.DEFAULT_HEIGHT,
            boxDepth: CONTAINERS.BOX.DEFAULT_DEPTH,
            cylinderRadius: CONTAINERS.CYLINDER.DEFAULT_RADIUS,
            cylinderHeight: CONTAINERS.CYLINDER.DEFAULT_HEIGHT,
            sphereRadius: CONTAINERS.SPHERE.DEFAULT_RADIUS,
            coneRadius: CONTAINERS.CONE.DEFAULT_RADIUS,
            coneHeight: CONTAINERS.CONE.DEFAULT_HEIGHT,
            coneTaper: CONTAINERS.CONE.DEFAULT_TAPER,
            venturiTopRadius: CONTAINERS.VENTURI.DEFAULT_TOP_RADIUS,
            venturiThroatRadius: CONTAINERS.VENTURI.DEFAULT_THROAT_RADIUS,
            venturiBottomRadius: CONTAINERS.VENTURI.DEFAULT_BOTTOM_RADIUS,
            venturiHeight: CONTAINERS.VENTURI.DEFAULT_HEIGHT,
            venturiThroatPosition: CONTAINERS.VENTURI.DEFAULT_THROAT_POSITION,
            venturiHelixCount: CONTAINERS.VENTURI.DEFAULT_HELIX_COUNT,
            venturiHelixPitch: CONTAINERS.VENTURI.DEFAULT_HELIX_PITCH,
            torusMajorRadius: CONTAINERS.TORUS.DEFAULT_MAJOR_RADIUS,
            torusMinorRadius: CONTAINERS.TORUS.DEFAULT_MINOR_RADIUS,
            torusHeight: CONTAINERS.TORUS.DEFAULT_HEIGHT,
            torusKnotCount: CONTAINERS.TORUS.DEFAULT_KNOT_COUNT,
            torusKnotIntensity: CONTAINERS.TORUS.DEFAULT_KNOT_INTENSITY,
            torusTwistAmount: CONTAINERS.TORUS.DEFAULT_TWIST_AMOUNT,
            chemotaxisEnabled: CONTAINERS.CHEMOTAXIS.DEFAULT_ENABLED,
            chemotaxisStrength: CONTAINERS.CHEMOTAXIS.DEFAULT_STRENGTH,
            chemotaxisDiffusionRate: CONTAINERS.CHEMOTAXIS.DEFAULT_DIFFUSION_RATE,
            chemotaxisDecayRate: CONTAINERS.CHEMOTAXIS.DEFAULT_DECAY_RATE,
            chemotaxisSourceRadius: CONTAINERS.CHEMOTAXIS.DEFAULT_SOURCE_RADIUS,
            chemotaxisAttraction: CONTAINERS.CHEMOTAXIS.DEFAULT_ATTRACTION,

            // Simulation state
            running: true,
        };
    }

    // Getters
    getParameters(): SimulationParameters {
        return { ...this.parameters };
    }

    getParameter<K extends keyof SimulationParameters>(key: K): SimulationParameters[K] {
        return this.parameters[key];
    }

    // Setters
    setParameter<K extends keyof SimulationParameters>(key: K, value: SimulationParameters[K]): void {
        this.parameters[key] = value;
    }

    setParameters(updates: Partial<SimulationParameters>): void {
        Object.assign(this.parameters, updates);
    }

    // Validation
    validateParameter<K extends keyof SimulationParameters>(key: K, value: SimulationParameters[K]): boolean {
        switch (key) {
            case 'gravity':
                return typeof value === 'number' && value >= -2.0 && value <= 2.0;
            case 'particleSize':
                return typeof value === 'number' && value >= 0.1 && value <= 2.0;
            case 'viscosity':
                return typeof value === 'number' && value >= 0.9 && value <= 1.0;
            case 'wallStiffness':
                return typeof value === 'number' && value >= 0.05 && value <= 0.5;
            case 'collisionDamping':
                return typeof value === 'number' && value >= 0.1 && value <= 1.0;
            case 'velocityCap':
                return typeof value === 'number' && value >= 5 && value <= 50;
            case 'shapeType':
                return SHAPE_OPTIONS.includes(value as string);
            case 'numParticles':
                return PARTICLE_OPTIONS.some(option => option.value === value);
            default:
                return true;
        }
    }

    // Reset methods
    resetToDefaults(): void {
        this.parameters = this.createDefaultParameters();
    }

    resetPhysics(): void {
        this.parameters.gravity = PHYSICS.DEFAULT_GRAVITY;
        this.parameters.particleSize = PHYSICS.DEFAULT_PARTICLE_SIZE;
        this.parameters.flipGravity = false;
        this.parameters.flipSpeed = PHYSICS.DEFAULT_FLIP_SPEED;
        this.parameters.viscosity = PHYSICS.DEFAULT_VISCOSITY;
        this.parameters.wallStiffness = PHYSICS.DEFAULT_WALL_STIFFNESS;
        this.parameters.collisionDamping = PHYSICS.DEFAULT_COLLISION_DAMPING;
        this.parameters.velocityCap = PHYSICS.DEFAULT_VELOCITY_CAP;
    }

    resetRendering(): void {
        this.parameters.sigma = RENDERING.DEFAULT_SIGMA;
        this.parameters.speed = RENDERING.DEFAULT_SPEED;
        this.parameters.colorDensity = RENDERING.DEFAULT_COLOR_DENSITY;
        this.parameters.r = RENDERING.DEFAULT_COLOR.r;
        this.parameters.g = RENDERING.DEFAULT_COLOR.g;
        this.parameters.b = RENDERING.DEFAULT_COLOR.b;
    }

    resetShapes(): void {
        this.parameters.shapeType = SHAPE_OPTIONS[1];
        this.parameters.boxWidth = CONTAINERS.BOX.DEFAULT_WIDTH;
        this.parameters.boxHeight = CONTAINERS.BOX.DEFAULT_HEIGHT;
        this.parameters.boxDepth = CONTAINERS.BOX.DEFAULT_DEPTH;
        this.parameters.cylinderRadius = CONTAINERS.CYLINDER.DEFAULT_RADIUS;
        this.parameters.cylinderHeight = CONTAINERS.CYLINDER.DEFAULT_HEIGHT;
        this.parameters.sphereRadius = CONTAINERS.SPHERE.DEFAULT_RADIUS;
        this.parameters.coneRadius = CONTAINERS.CONE.DEFAULT_RADIUS;
        this.parameters.coneHeight = CONTAINERS.CONE.DEFAULT_HEIGHT;
        this.parameters.coneTaper = CONTAINERS.CONE.DEFAULT_TAPER;
        this.parameters.venturiTopRadius = CONTAINERS.VENTURI.DEFAULT_TOP_RADIUS;
        this.parameters.venturiThroatRadius = CONTAINERS.VENTURI.DEFAULT_THROAT_RADIUS;
        this.parameters.venturiBottomRadius = CONTAINERS.VENTURI.DEFAULT_BOTTOM_RADIUS;
        this.parameters.venturiHeight = CONTAINERS.VENTURI.DEFAULT_HEIGHT;
        this.parameters.venturiThroatPosition = CONTAINERS.VENTURI.DEFAULT_THROAT_POSITION;
        this.parameters.venturiHelixCount = CONTAINERS.VENTURI.DEFAULT_HELIX_COUNT;
        this.parameters.venturiHelixPitch = CONTAINERS.VENTURI.DEFAULT_HELIX_PITCH;
        this.parameters.torusMajorRadius = CONTAINERS.TORUS.DEFAULT_MAJOR_RADIUS;
        this.parameters.torusMinorRadius = CONTAINERS.TORUS.DEFAULT_MINOR_RADIUS;
        this.parameters.torusHeight = CONTAINERS.TORUS.DEFAULT_HEIGHT;
        this.parameters.torusKnotCount = CONTAINERS.TORUS.DEFAULT_KNOT_COUNT;
        this.parameters.torusKnotIntensity = CONTAINERS.TORUS.DEFAULT_KNOT_INTENSITY;
        this.parameters.torusTwistAmount = CONTAINERS.TORUS.DEFAULT_TWIST_AMOUNT;
        this.parameters.chemotaxisEnabled = CONTAINERS.CHEMOTAXIS.DEFAULT_ENABLED;
        this.parameters.chemotaxisStrength = CONTAINERS.CHEMOTAXIS.DEFAULT_STRENGTH;
        this.parameters.chemotaxisDiffusionRate = CONTAINERS.CHEMOTAXIS.DEFAULT_DIFFUSION_RATE;
        this.parameters.chemotaxisDecayRate = CONTAINERS.CHEMOTAXIS.DEFAULT_DECAY_RATE;
        this.parameters.chemotaxisSourceRadius = CONTAINERS.CHEMOTAXIS.DEFAULT_SOURCE_RADIUS;
        this.parameters.chemotaxisAttraction = CONTAINERS.CHEMOTAXIS.DEFAULT_ATTRACTION;
    }

    // Utility methods
    getShapeTypeIndex(): number {
        return SHAPE_OPTIONS.indexOf(this.parameters.shapeType);
    }

    getParticleOption(): { label: string; value: number } | undefined {
        return PARTICLE_OPTIONS.find(option => option.value === this.parameters.numParticles);
    }

    // Event handlers
    toggleSimulation(): void {
        this.parameters.running = !this.parameters.running;
    }

    toggleFlipGravity(): void {
        this.parameters.flipGravity = !this.parameters.flipGravity;
    }
}
