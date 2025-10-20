struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
}
struct Cell {
    vx: i32, 
    vy: i32, 
    vz: i32, 
    mass: i32, 
}
override fixedPointMultiplier: f32; 

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<uniform> realBoxSize: vec3f;
@group(0) @binding(3) var<uniform> initBoxSize: vec3f;
@group(0) @binding(4) var<uniform> numParticles: u32;
@group(0) @binding(5) var<uniform> dt: f32;
@group(0) @binding(6) var<uniform> shapeType: u32;
@group(0) @binding(7) var<uniform> physicsProps: PhysicsProperties;
@group(0) @binding(8) var<uniform> shapeParams: ShapeParams;

struct PhysicsProperties {
    viscosity: f32,
    wallStiffness: f32,
    collisionDamping: f32,
    velocityCap: f32,
}

struct ShapeParams {
    boxWidth: f32,
    boxHeight: f32,
    boxDepth: f32,
    cylinderRadius: f32,
    cylinderHeight: f32,
    sphereRadius: f32,
    coneRadius: f32,
    coneHeight: f32,
    coneTaper: f32,
    padding: f32,
}

fn decodeFixedPoint(fixedPoint: i32) -> f32 {
    return f32(fixedPoint) / fixedPointMultiplier;
}


@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < numParticles) {
        particles[id.x].v = vec3f(0.);
        var weights: array<vec3f, 3>;

        let particle = particles[id.x];
        let cellIndex: vec3f = floor(particle.position);
        let cellDiff: vec3f = particle.position - (cellIndex + 0.5f);
        weights[0] = 0.5f * (0.5f - cellDiff) * (0.5f - cellDiff);
        weights[1] = 0.75f - cellDiff * cellDiff;
        weights[2] = 0.5f * (0.5f + cellDiff) * (0.5f + cellDiff);

        var B: mat3x3f = mat3x3f(vec3f(0.), vec3f(0.), vec3f(0.));
        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cellX: vec3f = vec3f(
                        cellIndex.x + f32(gx) - 1., 
                        cellIndex.y + f32(gy) - 1.,
                        cellIndex.z + f32(gz) - 1.  
                    );
                    let cellDist: vec3f = (cellX + 0.5f) - particle.position;
                    let cellIndex1D: i32 = 
                        i32(cellX.x) * i32(initBoxSize.y) * i32(initBoxSize.z) + 
                        i32(cellX.y) * i32(initBoxSize.z) + 
                        i32(cellX.z);
                    let weighted_velocity: vec3f = vec3f(
                        decodeFixedPoint(cells[cellIndex1D].vx), 
                        decodeFixedPoint(cells[cellIndex1D].vy), 
                        decodeFixedPoint(cells[cellIndex1D].vz)
                    ) * weight;
                    let term: mat3x3f = mat3x3f(
                        weighted_velocity * cellDist.x, 
                        weighted_velocity * cellDist.y, 
                        weighted_velocity * cellDist.z
                    );

                    B += term;

                    particles[id.x].v += weighted_velocity;
                }
            }
        }

        particles[id.x].C = B * 4.0f;
        
        // Apply single viscosity damping (removed redundant damping operations)
        particles[id.x].v *= physicsProps.viscosity;
        
        // Add velocity cap to prevent unrealistic speeds (configurable)
        let velocity_magnitude = length(particles[id.x].v);
        if (velocity_magnitude > physicsProps.velocityCap) {
            particles[id.x].v = normalize(particles[id.x].v) * physicsProps.velocityCap;
        }
        
        particles[id.x].position += particles[id.x].v * dt;
        
        // Apply boundary conditions (configurable)
        let wallStiffness = physicsProps.wallStiffness;
        let k = 2.0;
        let x_n = particles[id.x].position + particles[id.x].v * dt * k;
        
        // Apply boundary conditions based on shape type
        switch (shapeType) {
            case 0u: { // Box
                let center = realBoxSize * 0.5;
                let boxWidth = shapeParams.boxWidth;  // Use full width without clipping
                let boxHeight = shapeParams.boxHeight;  // Use full height without clipping
                let boxDepth = shapeParams.boxDepth;  // Use full depth without clipping
                let boxStartX = center.x - boxWidth * 0.5;
                let boxEndX = center.x + boxWidth * 0.5;
                let boxStartZ = center.z - boxDepth * 0.5;
                let boxEndZ = center.z + boxDepth * 0.5;
                
                if (x_n.x < boxStartX) { 
                    particles[id.x].v.x += wallStiffness * (boxStartX - x_n.x); 
                    particles[id.x].v.x *= physicsProps.collisionDamping;  // Configurable damping
                }
                if (x_n.x > boxEndX) { 
                    particles[id.x].v.x += wallStiffness * (boxEndX - x_n.x); 
                    particles[id.x].v.x *= physicsProps.collisionDamping;  // Configurable damping
                }
                if (x_n.y < 3.0) { 
                    particles[id.x].v.y += wallStiffness * (3.0 - x_n.y); 
                    particles[id.x].v.y *= physicsProps.collisionDamping * 0.7;  // Bottom damping (70% of normal)
                }
                if (x_n.y > boxHeight) { 
                    particles[id.x].v.y += wallStiffness * (boxHeight - x_n.y); 
                    particles[id.x].v.y *= physicsProps.collisionDamping;  // Configurable damping
                }
                if (x_n.z < boxStartZ) { 
                    particles[id.x].v.z += wallStiffness * (boxStartZ - x_n.z); 
                    particles[id.x].v.z *= physicsProps.collisionDamping;  // Configurable damping
                }
                if (x_n.z > boxEndZ) { 
                    particles[id.x].v.z += wallStiffness * (boxEndZ - x_n.z); 
                    particles[id.x].v.z *= physicsProps.collisionDamping;  // Configurable damping
                }
            }
            case 1u: { // Cylinder
                let center = realBoxSize * 0.5;
                let radius = shapeParams.cylinderRadius;  // Use full radius without clipping
                let height = shapeParams.cylinderHeight;  // Use full height without clipping
                let dist_from_center = length(x_n.xz - center.xz);
                if (dist_from_center > radius) {
                    let normal = normalize(x_n.xz - center.xz);
                    let penetration = dist_from_center - radius;
                    particles[id.x].v.x += wallStiffness * normal.x * penetration;
                    particles[id.x].v.z += wallStiffness * normal.y * penetration;
                    particles[id.x].v.x *= physicsProps.collisionDamping;  // Configurable damping
                    particles[id.x].v.z *= physicsProps.collisionDamping;  // Configurable damping
                }
                if (x_n.y < 3.0) { 
                    particles[id.x].v.y += wallStiffness * (3.0 - x_n.y); 
                    particles[id.x].v.y *= physicsProps.collisionDamping * 0.5;  // Bottom damping (50% of normal)
                }
                if (x_n.y > height) { 
                    particles[id.x].v.y += wallStiffness * (height - x_n.y); 
                    particles[id.x].v.y *= physicsProps.collisionDamping;  // Configurable damping
                }
            }
            case 2u: { // Sphere
                let center = realBoxSize * 0.5;
                let radius = shapeParams.sphereRadius;
                let dist_from_center = length(x_n - center);
                
                if (dist_from_center > radius) {
                    let normal = normalize(x_n - center);
                    let penetration = dist_from_center - radius;
                    particles[id.x].v += wallStiffness * normal * penetration;
                    particles[id.x].v *= physicsProps.collisionDamping;
                }
            }
            case 3u: { // Cone
                let center = realBoxSize * 0.5;
                let radius = min(realBoxSize.x, realBoxSize.z) * 0.4;
                let height_factor = x_n.y / realBoxSize.y;
                let current_radius = radius * height_factor;
                let dist_from_center = length(x_n.xz - center.xz);
                if (dist_from_center > current_radius) {
                    let normal = normalize(x_n.xz - center.xz);
                    let penetration = dist_from_center - current_radius;
                    particles[id.x].v.x += wallStiffness * normal.x * penetration;
                    particles[id.x].v.z += wallStiffness * normal.y * penetration;
                    particles[id.x].v.x *= physicsProps.collisionDamping;  // Configurable damping
                    particles[id.x].v.z *= physicsProps.collisionDamping;  // Configurable damping
                }
                if (x_n.y < 3.0) { 
                    particles[id.x].v.y += wallStiffness * (3.0 - x_n.y); 
                    particles[id.x].v.y *= physicsProps.collisionDamping * 0.5;  // Bottom damping (50% of normal)
                }
                if (x_n.y > realBoxSize.y - 4.0) { 
                    particles[id.x].v.y += wallStiffness * (realBoxSize.y - 4.0 - x_n.y); 
                    particles[id.x].v.y *= physicsProps.collisionDamping;  // Configurable damping
                }
            }
            default: { // Default to box
                if (x_n.x < 3.0) { 
                    particles[id.x].v.x += wallStiffness * (3.0 - x_n.x); 
                    particles[id.x].v.x *= physicsProps.collisionDamping;  // Configurable damping
                }
                if (x_n.x > realBoxSize.x - 4.0) { 
                    particles[id.x].v.x += wallStiffness * (realBoxSize.x - 4.0 - x_n.x); 
                    particles[id.x].v.x *= physicsProps.collisionDamping;  // Configurable damping
                }
                if (x_n.y < 3.0) { 
                    particles[id.x].v.y += wallStiffness * (3.0 - x_n.y); 
                    particles[id.x].v.y *= physicsProps.collisionDamping * 0.5;  // Bottom damping (50% of normal)
                }
                if (x_n.y > realBoxSize.y - 4.0) { 
                    particles[id.x].v.y += wallStiffness * (realBoxSize.y - 4.0 - x_n.y); 
                    particles[id.x].v.y *= physicsProps.collisionDamping;  // Configurable damping
                }
                if (x_n.z < 3.0) { 
                    particles[id.x].v.z += wallStiffness * (3.0 - x_n.z); 
                    particles[id.x].v.z *= physicsProps.collisionDamping;  // Configurable damping
                }
                if (x_n.z > realBoxSize.z - 4.0) { 
                    particles[id.x].v.z += wallStiffness * (realBoxSize.z - 4.0 - x_n.z); 
                    particles[id.x].v.z *= physicsProps.collisionDamping;  // Configurable damping
                }
            }
        }
    }
}