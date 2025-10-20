import { PHYSICS, SIMULATION } from '../utils/Constants';
import { UniformManager } from '../buffers/UniformManager';

export interface PhysicsProperties {
    viscosity: number;
    wallStiffness: number;
    collisionDamping: number;
    velocityCap: number;
}

export class PhysicsEngine {
    private uniformManager: UniformManager;
    private properties: PhysicsProperties;

    constructor(uniformManager: UniformManager) {
        this.uniformManager = uniformManager;
        this.properties = {
            viscosity: PHYSICS.DEFAULT_VISCOSITY,
            wallStiffness: PHYSICS.DEFAULT_WALL_STIFFNESS,
            collisionDamping: PHYSICS.DEFAULT_COLLISION_DAMPING,
            velocityCap: PHYSICS.DEFAULT_VELOCITY_CAP,
        };
    }

    // Getters
    getProperties(): PhysicsProperties {
        return { ...this.properties };
    }

    getViscosity(): number {
        return this.properties.viscosity;
    }

    getWallStiffness(): number {
        return this.properties.wallStiffness;
    }

    getCollisionDamping(): number {
        return this.properties.collisionDamping;
    }

    getVelocityCap(): number {
        return this.properties.velocityCap;
    }

    // Setters
    setViscosity(viscosity: number): void {
        this.properties.viscosity = Math.max(0.9, Math.min(1.0, viscosity));
        this.updateUniforms();
    }

    setWallStiffness(wallStiffness: number): void {
        this.properties.wallStiffness = Math.max(0.05, Math.min(0.5, wallStiffness));
        this.updateUniforms();
    }

    setCollisionDamping(collisionDamping: number): void {
        this.properties.collisionDamping = Math.max(0.1, Math.min(1.0, collisionDamping));
        this.updateUniforms();
    }

    setVelocityCap(velocityCap: number): void {
        this.properties.velocityCap = Math.max(5, Math.min(50, velocityCap));
        this.updateUniforms();
    }

    setProperties(properties: Partial<PhysicsProperties>): void {
        if (properties.viscosity !== undefined) {
            this.setViscosity(properties.viscosity);
        }
        if (properties.wallStiffness !== undefined) {
            this.setWallStiffness(properties.wallStiffness);
        }
        if (properties.collisionDamping !== undefined) {
            this.setCollisionDamping(properties.collisionDamping);
        }
        if (properties.velocityCap !== undefined) {
            this.setVelocityCap(properties.velocityCap);
        }
    }

    // Reset to defaults
    resetToDefaults(): void {
        this.properties = {
            viscosity: PHYSICS.DEFAULT_VISCOSITY,
            wallStiffness: PHYSICS.DEFAULT_WALL_STIFFNESS,
            collisionDamping: PHYSICS.DEFAULT_COLLISION_DAMPING,
            velocityCap: PHYSICS.DEFAULT_VELOCITY_CAP,
        };
        this.updateUniforms();
    }

    // Update GPU uniforms
    private updateUniforms(): void {
        this.uniformManager.updatePhysicsProperties(
            this.properties.viscosity,
            this.properties.wallStiffness,
            this.properties.collisionDamping,
            this.properties.velocityCap
        );
    }

    // Physics calculations
    calculateDamping(): number {
        return 0.999; // Global damping factor
    }

    calculateFlowSmoothing(): number {
        return 0.98; // Flow smoothing factor
    }

    calculateViscosityEffect(): number {
        return this.properties.viscosity;
    }

    // Validation methods
    validateViscosity(viscosity: number): boolean {
        return viscosity >= 0.9 && viscosity <= 1.0;
    }

    validateWallStiffness(stiffness: number): boolean {
        return stiffness >= 0.05 && stiffness <= 0.5;
    }

    validateCollisionDamping(damping: number): boolean {
        return damping >= 0.1 && damping <= 1.0;
    }

    validateVelocityCap(cap: number): boolean {
        return cap >= 5 && cap <= 50;
    }
}

