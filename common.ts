import { UniformManager } from './buffers/UniformManager';

// Singleton for migration period
let globalUniformManager: UniformManager;

export function initCommonGlobals(uniformManager: UniformManager) {
    globalUniformManager = uniformManager;
    // Update the exports to point to the manager's values
    renderUniformsValues = uniformManager.renderUniformsValues;
    renderUniformsViews = uniformManager.renderUniformsViews;
    physicsUniformsValues = uniformManager.physicsUniformsValues;
    physicsUniformsViews = uniformManager.physicsUniformsViews;
}

// Legacy fallback values for backward compatibility
const legacyRenderUniformsValues = new ArrayBuffer(288);
const legacyRenderUniformsViews = {
  texel_size: new Float32Array(legacyRenderUniformsValues, 0, 2),
  sphere_size: new Float32Array(legacyRenderUniformsValues, 8, 2),
  inv_projection_matrix: new Float32Array(legacyRenderUniformsValues, 16, 16),
  projection_matrix: new Float32Array(legacyRenderUniformsValues, 80, 16),
  view_matrix: new Float32Array(legacyRenderUniformsValues, 144, 16),
  inv_view_matrix: new Float32Array(legacyRenderUniformsValues, 208, 16),
  gravity: new Float32Array(legacyRenderUniformsValues, 272, 1),
};

const legacyPhysicsUniformsValues = new ArrayBuffer(16);
const legacyPhysicsUniformsViews = {
  viscosity: new Float32Array(legacyPhysicsUniformsValues, 0, 1),
  wallStiffness: new Float32Array(legacyPhysicsUniformsValues, 4, 1),
  collisionDamping: new Float32Array(legacyPhysicsUniformsValues, 8, 1),
  velocityCap: new Float32Array(legacyPhysicsUniformsValues, 12, 1),
};

// Initialize default physics values
legacyPhysicsUniformsViews.viscosity[0] = 0.995;
legacyPhysicsUniformsViews.wallStiffness[0] = 0.2;
legacyPhysicsUniformsViews.collisionDamping[0] = 0.7;
legacyPhysicsUniformsViews.velocityCap[0] = 25.0;

// Re-export from manager for backward compatibility, with fallback
// Note: These will be updated when initCommonGlobals is called
export let renderUniformsValues = legacyRenderUniformsValues;
export let renderUniformsViews = legacyRenderUniformsViews;
export let physicsUniformsValues = legacyPhysicsUniformsValues;
export let physicsUniformsViews = legacyPhysicsUniformsViews;

export const numParticlesMax = 1600000;

// Add deprecation warnings in dev mode
if (import.meta.env.DEV) {
    console.warn('common.ts globals are deprecated. Use UniformManager directly.');
}
