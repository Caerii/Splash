// Simulation Constants
export const SIMULATION = {
    MAX_PARTICLES: 1600000,
    PARTICLE_STRUCT_SIZE: 80,
    CELL_STRUCT_SIZE: 16,
    MAX_GRID_COUNT: 120 * 120 * 120,
    FIXED_POINT_MULTIPLIER: 1000000,
    DEFAULT_DT: 0.016, // ~60 FPS
} as const;

// Physics Constants
export const PHYSICS = {
    DEFAULT_GRAVITY: -0.40,
    DEFAULT_VISCOSITY: 0.995,
    DEFAULT_WALL_STIFFNESS: 0.3, // Increased for more responsive boundaries
    DEFAULT_COLLISION_DAMPING: 0.9, // Reduced from 0.7 to 0.9 for less sticking
    DEFAULT_VELOCITY_CAP: 25.0,
    DEFAULT_PARTICLE_SIZE: 0.9,
    DEFAULT_FLIP_SPEED: 0.5,
    MOUSE_FORCE_STRENGTH: 0.2,
    MOUSE_RADIUS: 140,
} as const;

// Rendering Constants
export const RENDERING = {
    DEFAULT_TEXEL_SIZE: [1.0, 1.0],
    DEFAULT_SPHERE_SIZE: [0.5, 0.5],
    DEFAULT_COLOR_DENSITY: 3.0,
    DEFAULT_SIGMA: 1.3,
    DEFAULT_SPEED: 0.8,
    DEFAULT_COLOR: { r: 140, g: 220, b: 240 },
} as const;

// Container Shape Constants
export const CONTAINERS = {
    BOX: {
        DEFAULT_WIDTH: 60,
        DEFAULT_HEIGHT: 50,
        DEFAULT_DEPTH: 60,
        MIN_SIZE: 20,
        MAX_SIZE: 100,
    },
    CYLINDER: {
        DEFAULT_RADIUS: 30,
        DEFAULT_HEIGHT: 50,
        MIN_RADIUS: 10,
        MAX_RADIUS: 50,
    },
    SPHERE: {
        DEFAULT_RADIUS: 25,
        MIN_RADIUS: 10,
        MAX_RADIUS: 40,
    },
    CONE: {
        DEFAULT_RADIUS: 30,
        DEFAULT_HEIGHT: 50,
        DEFAULT_TAPER: 0.5,
        MIN_RADIUS: 10,
        MAX_RADIUS: 50,
        MIN_HEIGHT: 20,
        MAX_HEIGHT: 80,
    },
} as const;

// Particle Count Options
export const PARTICLE_OPTIONS = [
    { label: '10K', value: 10000 },
    { label: '50K', value: 50000 },
    { label: '100K', value: 100000 },
    { label: '200K', value: 200000 },
    { label: '500K', value: 500000 },
    { label: '1M', value: 1000000 },
];

// Shape Options
export const SHAPE_OPTIONS = ['Box', 'Cylinder', 'Sphere', 'Cone'];

// Buffer Sizes
export const BUFFER_SIZES = {
    RENDER_UNIFORMS: 288,
    PHYSICS_UNIFORMS: 16,
    MOUSE_INFO: 32,
    DT_BUFFER: 4,
    SHAPE_TYPE: 4,
    NUM_PARTICLES: 4,
    SHAPE_PARAMS: 40, // 10 f32 values for all shape parameters
} as const;

// WebGPU Limits
export const WEBGPU_LIMITS = {
    MAX_BINDINGS_PER_GROUP: 8,
    MAX_BUFFER_SIZE: 268435456, // 256MB
    MAX_TEXTURE_SIZE: 8192,
} as const;

// Camera Constants
export const CAMERA = {
    DEFAULT_FOV: 45,
    DEFAULT_ZOOM_RATE: 1.0,
    DEFAULT_DISTANCE: 200,
    DEFAULT_TARGET_Y: 25,
    MIN_DISTANCE: 50,
    MAX_DISTANCE: 500,
} as const;

// Mouse Interaction Constants
export const MOUSE = {
    DEFAULT_RADIUS: 140,
    DEFAULT_FORCE: 0.2,
    MIN_RADIUS: 50,
    MAX_RADIUS: 300,
    MIN_FORCE: 0.0,
    MAX_FORCE: 1.0,
} as const;

// GUI Constants
export const GUI = {
    FOLDER_NAMES: {
        PHYSICS: 'Physics',
        SHAPES: 'Container Shape',
        COLORS: 'Colors',
        PARTICLES: 'Number of Particles',
        SPEED: 'Speed',
    },
    CONTROL_RANGES: {
        GRAVITY: { min: -2.0, max: 2.0, step: 0.01 },
        PARTICLE_SIZE: { min: 0.1, max: 2.0, step: 0.01 },
        FLIP_SPEED: { min: 0.1, max: 2.0, step: 0.1 },
        VISCOSITY: { min: 0.9, max: 1.0, step: 0.001 },
        WALL_STIFFNESS: { min: 0.05, max: 0.5, step: 0.01 },
        COLLISION_DAMPING: { min: 0.1, max: 1.0, step: 0.01 },
        VELOCITY_CAP: { min: 5, max: 50, step: 1 },
    },
} as const;

