struct Cell {
    vx: i32, 
    vy: i32, 
    vz: i32, 
    mass: i32, 
}
struct RenderUniforms {
    texelSize: vec2f, 
    sphereSize: f32, 
    invProjectionMatrix: mat4x4f, 
    projectionMatrix: mat4x4f, 
    viewMatrix: mat4x4f, 
    invViewMatrix: mat4x4f, 
    gravity: f32, 
}
struct MouseInfo {
    screenSize: vec2f, 
    mouseCoord : vec2f, 
    mouseVel : vec2f, 
    mouseRadius: f32, 
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
    venturiTopRadius: f32,
    venturiThroatRadius: f32,
    venturiBottomRadius: f32,
    venturiHeight: f32,
    venturiThroatPosition: f32,
    venturiHelixCount: f32,
    venturiHelixPitch: f32,
    torusMajorRadius: f32,
    torusMinorRadius: f32,
    torusHeight: f32,
    torusKnotCount: f32,
    torusKnotIntensity: f32,
    torusTwistAmount: f32,
    chemotaxisEnabled: f32,
    chemotaxisStrength: f32,
    chemotaxisDiffusionRate: f32,
    chemotaxisDecayRate: f32,
    chemotaxisSourceRadius: f32,
    chemotaxisAttraction: f32,
    padding: f32,
}

override fixedPointMultiplier: f32; 

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> realBoxSize: vec3f;
@group(0) @binding(2) var<uniform> initBoxSize: vec3f;
@group(0) @binding(3) var<uniform> uniforms: RenderUniforms;
@group(0) @binding(4) var depthTexture: texture_2d<f32>;
@group(0) @binding(5) var<uniform> mouseInfo: MouseInfo;
@group(0) @binding(6) var<uniform> dt: f32;
@group(0) @binding(7) var<uniform> shapeType: u32;
@group(0) @binding(8) var<uniform> shapeParams: ShapeParams;

fn encodeFixedPoint(floatingPoint: f32) -> i32 {
    return i32(floatingPoint * fixedPointMultiplier);
}
fn decodeFixedPoint(fixedPoint: i32) -> f32 {
    return f32(fixedPoint) / fixedPointMultiplier;
}

fn computeViewPosFromUVDepth(tex_coord: vec2f, depth: f32) -> vec3f {
    var ndc: vec4f = vec4f(tex_coord.x * 2.0 - 1.0, 1.0 - 2.0 * tex_coord.y, 0.0, 1.0);
    ndc.z = -uniforms.projectionMatrix[2].z + uniforms.projectionMatrix[3].z / depth;
    ndc.w = 1.0;

    var eye_pos: vec4f = uniforms.invProjectionMatrix * ndc;

    return eye_pos.xyz / eye_pos.w;
}

fn getViewPosFromTexCoord(tex_coord: vec2f, iuv: vec2f) -> vec3f {
    var depth: f32 = abs(textureLoad(depthTexture, vec2u(iuv), 0).x);
    return computeViewPosFromUVDepth(tex_coord, depth);
}

@compute @workgroup_size(64)
fn updateGrid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&cells)) { // TODO : 変える
        let uv: vec2f = mouseInfo.mouseCoord;
        let iuv = uv * mouseInfo.screenSize;
        let depth: f32 = abs(textureLoad(depthTexture, vec2u(iuv), 0).x);
        var mouseCellIndex: u32 = 1000000000; // 適当な invalid 値
        var cellSquareDistToMouse: f32 = 1e9;
        var forceDir = vec3f(0.);

        if (depth < 1e4) {
            let mouseViewPos = getViewPosFromTexCoord(uv, iuv);
            let mouseWorldPos = uniforms.invViewMatrix * vec4f(mouseViewPos, 1.); // 位置なので 1
            let mouseCellPos: vec3i = vec3i(floor(mouseWorldPos).xyz);
            mouseCellIndex =    u32(mouseCellPos.x) * u32(initBoxSize.y) * u32(initBoxSize.z) + 
                                u32(mouseCellPos.y) * u32(initBoxSize.z) + 
                                u32(mouseCellPos.z);
            let center = realBoxSize / 2;
            forceDir = select(vec3f(0.), (uniforms.invViewMatrix * vec4f(mouseInfo.mouseVel, 0.0, 0)).xyz, dot(mouseInfo.mouseVel, mouseInfo.mouseVel) > 0.);
            var x: f32 = f32(i32(id.x) / i32(initBoxSize.z) / i32(initBoxSize.y));
            var y: f32 = f32((i32(id.x) / i32(initBoxSize.z)) % i32(initBoxSize.y));
            var z: f32 = f32(i32(id.x) % i32(initBoxSize.z));
            let cellPos = vec3f(x, y, z);
            let diff = floor(mouseWorldPos).xyz - cellPos;
            cellSquareDistToMouse = dot(diff, diff);
        }

        let dt = dt;
        let r = mouseInfo.mouseRadius;

        // Early exit for cells with no mass - avoid unnecessary calculations
        if (cells[id.x].mass <= 0) {
            return;
        }
        
        var floatV: vec3f = vec3f(
            decodeFixedPoint(cells[id.x].vx), 
            decodeFixedPoint(cells[id.x].vy), 
            decodeFixedPoint(cells[id.x].vz)
        );
        floatV /= decodeFixedPoint(cells[id.x].mass);

        // Constants for better maintainability
        const MOUSE_FORCE_STRENGTH = 0.2;
        const MIN_BOUNDARY_DISTANCE = 2;
        
        let strength = smoothstep(r*r, 0., cellSquareDistToMouse) * MOUSE_FORCE_STRENGTH;   
        cells[id.x].vx = encodeFixedPoint(floatV.x + strength * forceDir.x); 
        cells[id.x].vy = encodeFixedPoint(floatV.y + strength * forceDir.y + uniforms.gravity * dt); 
        cells[id.x].vz = encodeFixedPoint(floatV.z + strength * forceDir.z); 

        var x: i32 = i32(id.x) / i32(initBoxSize.z) / i32(initBoxSize.y);
        var y: i32 = (i32(id.x) / i32(initBoxSize.z)) % i32(initBoxSize.y);
        var z: i32 = i32(id.x) % i32(initBoxSize.z);
        
        // Apply boundary conditions based on shape type
        switch (shapeType) {
            case 0u: { // Box
                let center = realBoxSize * 0.5;
                let boxWidth = shapeParams.boxWidth;
                let boxHeight = shapeParams.boxHeight;
                let boxDepth = shapeParams.boxDepth;
                let boxStartX = center.x - boxWidth * 0.5;
                let boxEndX = center.x + boxWidth * 0.5;
                let boxStartZ = center.z - boxDepth * 0.5;
                let boxEndZ = center.z + boxDepth * 0.5;
                
                // Optimized boundary checks - avoid redundant f32() conversions
                let fx = f32(x);
                let fy = f32(y);
                let fz = f32(z);
                
                if (fx < boxStartX || fx > boxEndX) { cells[id.x].vx = 0; }
                if (y < MIN_BOUNDARY_DISTANCE || fy > boxHeight) { cells[id.x].vy = 0; }
                if (fz < boxStartZ || fz > boxEndZ) { cells[id.x].vz = 0; }
            }
            case 1u: { // Cylinder
                let center = realBoxSize * 0.5;
                let radius = shapeParams.cylinderRadius;
                let height = shapeParams.cylinderHeight;
                let cell_pos = vec3f(f32(x), f32(y), f32(z));
                let dist_from_center = length(cell_pos.xz - center.xz);
                if (dist_from_center > radius) { 
                    cells[id.x].vx = 0; 
                    cells[id.x].vz = 0; 
                }
                if (y < 2 || f32(y) > height) { cells[id.x].vy = 0; }
            }
            case 2u: { // Sphere
                let center = realBoxSize * 0.5;
                let radius = shapeParams.sphereRadius;
                let cell_pos = vec3f(f32(x), f32(y), f32(z));
                let dist_from_center = length(cell_pos - center);
                
                if (dist_from_center > radius) { 
                    cells[id.x].vx = 0; 
                    cells[id.x].vy = 0; 
                    cells[id.x].vz = 0; 
                }
            }
            case 3u: { // Cone
                let center = realBoxSize * 0.5;
                let radius = min(realBoxSize.x, realBoxSize.z) * 0.4;
                let cell_pos = vec3f(f32(x), f32(y), f32(z));
                let height_factor = cell_pos.y / realBoxSize.y;
                let current_radius = radius * height_factor;
                let dist_from_center = length(cell_pos.xz - center.xz);
                if (dist_from_center > current_radius) { 
                    cells[id.x].vx = 0; 
                    cells[id.x].vz = 0; 
                }
                if (y < 2 || y > i32(ceil(realBoxSize.y) - 3)) { cells[id.x].vy = 0; }
            }
            case 4u: { // Triple Helix Venturi
                let center = realBoxSize * 0.5;
                let topRadius = shapeParams.venturiTopRadius;
                let throatRadius = shapeParams.venturiThroatRadius;
                let bottomRadius = shapeParams.venturiBottomRadius;
                let height = shapeParams.venturiHeight;
                let throatPosition = shapeParams.venturiThroatPosition;
                let helixCount = shapeParams.venturiHelixCount;
                let helixPitch = shapeParams.venturiHelixPitch;
                let cell_pos = vec3f(f32(x), f32(y), f32(z));
                
                // Calculate radius at current cell height using smooth interpolation
                let t = cell_pos.y / height;
                var currentRadius: f32;
                if (t < throatPosition) {
                    let localT = t / throatPosition;
                    let eased = 1.0 - pow(1.0 - localT, 3.0); // ease-out cubic
                    currentRadius = topRadius + (throatRadius - topRadius) * eased;
                } else {
                    let localT = (t - throatPosition) / (1.0 - throatPosition);
                    let eased = pow(localT, 3.0); // ease-in cubic
                    currentRadius = throatRadius + (bottomRadius - throatRadius) * eased;
                }
                
                // Check if cell is within any helix strand
                var is_inside_helix = false;
                
                for (var helix = 0; helix < i32(helixCount); helix++) {
                    let helixAngle = f32(helix) / helixCount * 2.0 * 3.14159;
                    let helixOffset = t * helixPitch * 2.0 * 3.14159;
                    let currentHelixAngle = helixAngle + helixOffset;
                    
                    // Calculate distance to this helix strand
                    let helixCenterX = center.x + currentRadius * 0.6 * cos(currentHelixAngle);
                    let helixCenterZ = center.z + currentRadius * 0.6 * sin(currentHelixAngle);
                    let helixRadius = currentRadius * 0.4;
                    
                    let dist_to_helix = length(cell_pos.xz - vec2f(helixCenterX, helixCenterZ));
                    if (dist_to_helix <= helixRadius) {
                        is_inside_helix = true;
                        break;
                    }
                }
                
                // If cell is outside all helix strands, zero out velocities
                if (!is_inside_helix) { 
                    cells[id.x].vx = 0; 
                    cells[id.x].vz = 0; 
                }
                if (y < 2 || f32(y) > height) { cells[id.x].vy = 0; }
            }
            case 5u: { // Knotted Torus
                let center = realBoxSize * 0.5;
                let majorRadius = shapeParams.torusMajorRadius;
                let minorRadius = shapeParams.torusMinorRadius;
                let height = shapeParams.torusHeight;
                let knotCount = shapeParams.torusKnotCount;
                let knotIntensity = shapeParams.torusKnotIntensity;
                let twistAmount = shapeParams.torusTwistAmount;
                let cell_pos = vec3f(f32(x), f32(y), f32(z));
                
                // Calculate knotted torus position
                let dist_from_center = length(cell_pos.xz - center.xz);
                let angle = atan2(cell_pos.z - center.z, cell_pos.x - center.x);
                let height_factor = cell_pos.y / height;
                
                // Apply knotting based on knot count
                var knot_offset = 0.0;
                if (knotCount > 0.0) {
                    let knot_angle = angle * knotCount + height_factor * twistAmount * 6.28318;
                    knot_offset = sin(knot_angle) * knotIntensity * minorRadius * 0.5;
                }
                
                // Calculate the effective major radius with knotting
                let effective_major_radius = majorRadius + knot_offset;
                let dist_from_torus_center = abs(dist_from_center - effective_major_radius);
                
                // If cell is outside the knotted torus, zero out velocities
                if (dist_from_torus_center > minorRadius) { 
                    cells[id.x].vx = 0; 
                    cells[id.x].vz = 0; 
                }
                if (y < 2 || f32(y) > height) { cells[id.x].vy = 0; }
            }
            default: { // Default to box
                if (x < 2 || x > i32(ceil(realBoxSize.x) - 3)) { cells[id.x].vx = 0; }
                if (y < 2 || y > i32(ceil(realBoxSize.y) - 3)) { cells[id.x].vy = 0; }
                if (z < 2 || z > i32(ceil(realBoxSize.z) - 3)) { cells[id.x].vz = 0; }
            }
        }
    }
}