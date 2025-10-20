export const renderUniformsValues = new ArrayBuffer(288);
export const renderUniformsViews = {
  texel_size: new Float32Array(renderUniformsValues, 0, 2),
  sphere_size: new Float32Array(renderUniformsValues, 8, 2),
  inv_projection_matrix: new Float32Array(renderUniformsValues, 16, 16),
  projection_matrix: new Float32Array(renderUniformsValues, 80, 16),
  view_matrix: new Float32Array(renderUniformsValues, 144, 16),
  inv_view_matrix: new Float32Array(renderUniformsValues, 208, 16),
  gravity: new Float32Array(renderUniformsValues, 272, 1),
};

// Physics properties uniform buffer (16 bytes for 4 floats)
export const physicsUniformsValues = new ArrayBuffer(16);
export const physicsUniformsViews = {
  viscosity: new Float32Array(physicsUniformsValues, 0, 1),      // 0.995
  wallStiffness: new Float32Array(physicsUniformsValues, 4, 1),  // 0.2
  collisionDamping: new Float32Array(physicsUniformsValues, 8, 1), // 0.7
  velocityCap: new Float32Array(physicsUniformsValues, 12, 1),   // 25.0
};

export const numParticlesMax = 1600000;

// Initialize default physics values
physicsUniformsViews.viscosity[0] = 0.995;
physicsUniformsViews.wallStiffness[0] = 0.2;
physicsUniformsViews.collisionDamping[0] = 0.7;
physicsUniformsViews.velocityCap[0] = 25.0;
