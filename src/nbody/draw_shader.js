// compute_nbody_collisions (native_ts_graphics), the rendering in WGSL: the balls, their trails, the
// coordinate planes, the axis gizmo, the bloom's passes and the composite into the back buffer.
// Ported from shaders/compute_nbody_collisions.hlsl and compute_nbody_collisions_post.hlsl.
//
// The matrices are Donut's, row-major for row vectors (mul(v, M)); uploaded as they are, WGSL
// reads them as the transpose, so mul(v, M) is M * v here.
const DRAW_WGSL = `
struct Particle {
    pos: vec4<f32>,
    vel: vec4<f32>,
    info: vec4<f32>,
    spin: vec4<f32>,
    marker: vec4<f32>,
};

struct ViewConstants {
    worldToView: mat4x4<f32>,
    viewToClip: mat4x4<f32>,
    viewportSizeInv: vec2<f32>,
    pad: vec2<f32>,
};

const GIZMO_MAX_SEGMENTS = 16;
struct GizmoConstants {
    header: vec4<f32>,                              // xy = 1 / viewport size
    points: array<vec4<f32>, GIZMO_MAX_SEGMENTS>,   // x0, y0, x1, y1
    colors: array<vec4<f32>, GIZMO_MAX_SEGMENTS>,
};

@group(0) @binding(0) var<uniform> g_View: ViewConstants;
@group(0) @binding(1) var<storage, read> t_Particles: array<Particle>;
// x = the world distance a unit of velocity stretches to, y = the trails' opacity at the head,
// z = 1: the balls are lit spheres, w = 1: the balls show their spin marks.
@group(0) @binding(2) var<uniform> g_Trail: vec4<f32>;
@group(0) @binding(3) var<uniform> g_Gizmo: GizmoConstants;

// The sprite's half size, in collision radii: the solid ball fills the inner half, its glow the rest.
const SPRITE_EXTENT = 2.0;

var<private> c_QuadCorners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, -1.0));
var<private> c_GizmoQuadIndex = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);

struct BallVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) spriteCoord: vec2<f32>,
    @location(1) gradientPos: f32,
    @location(2) heat: f32,
    @location(3) marker: vec3<f32>,
    @location(4) lit: f32,
};

@vertex
fn main_vs(@builtin(vertex_index) vertexIndex: u32) -> BallVertex {
    var out: BallVertex;
    let particle = t_Particles[vertexIndex / 6u];
    let corner = c_QuadCorners[vertexIndex % 6u];
    out.lit = g_Trail.z;
    out.spriteCoord = vec2<f32>(0.0, 0.0);
    out.gradientPos = 0.0;
    out.heat = 0.0;
    out.marker = vec3<f32>(0.0, 0.0, 0.0);
    if (particle.info.x < 0.0) {
        // Merged into another particle: a degenerate quad, off screen.
        out.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        return out;
    }

    // Sized in the world so that the balls touch on screen when they collide. At least a pixel
    // across, so distant ones don't flicker. By the true radius (marker.w): a merged particle in a
    // pile collides smaller until it grows back, but it shows its size.
    let eyePos = g_View.worldToView * vec4<f32>(particle.pos.xyz, 1.0);
    var clip = g_View.viewToClip * eyePos;
    let halfSize = SPRITE_EXTENT * particle.marker.w;
    let ndcHalfSize = max(halfSize * vec2<f32>(g_View.viewToClip[0][0], g_View.viewToClip[1][1]) / clip.w,
        2.0 * g_View.viewportSizeInv);
    clip = vec4<f32>(clip.xy + corner * ndcHalfSize * clip.w, clip.zw);
    out.position = clip;

    out.spriteCoord = corner;
    out.gradientPos = particle.vel.w;
    out.heat = particle.info.y;
    // No marks: a zero marker is at no pole.
    if (g_Trail.w > 0.5) {
        out.marker = (g_View.worldToView * vec4<f32>(particle.marker.xyz, 0.0)).xyz;
    }
    return out;
}

// compute_nbody's textures (particle_gradient_rgba.ktx at every 16th texel, and particle_rgba.ktx
// averaged at 17 distances from the center), as tables.
var<private> c_Gradient = array<vec3<f32>, 17>(
    vec3<f32>(0.000, 0.663, 0.878), vec3<f32>(0.027, 0.541, 0.792), vec3<f32>(0.075, 0.369, 0.675),
    vec3<f32>(0.169, 0.224, 0.580), vec3<f32>(0.357, 0.145, 0.545), vec3<f32>(0.624, 0.106, 0.557),
    vec3<f32>(0.855, 0.090, 0.553), vec3<f32>(0.922, 0.086, 0.482), vec3<f32>(0.922, 0.086, 0.357),
    vec3<f32>(0.922, 0.118, 0.231), vec3<f32>(0.933, 0.247, 0.176), vec3<f32>(0.992, 0.510, 0.176),
    vec3<f32>(0.992, 0.788, 0.176), vec3<f32>(0.957, 0.914, 0.180), vec3<f32>(0.655, 0.867, 0.227),
    vec3<f32>(0.275, 0.722, 0.286), vec3<f32>(0.012, 0.620, 0.329));
var<private> c_Glow = array<f32, 17>(
    0.931, 0.826, 0.633, 0.484, 0.377, 0.290, 0.241, 0.207, 0.175,
    0.150, 0.132, 0.111, 0.098, 0.079, 0.054, 0.030, 0.008);

fn SampleGradient(u: f32) -> vec3<f32> {
    let x = saturate(u) * 16.0;
    let i = min(i32(x), 15);
    return mix(c_Gradient[i], c_Gradient[i + 1], x - f32(i));
}

fn SampleGlow(r: f32) -> f32 {
    if (r >= 1.0) {
        return 0.0;
    }
    let x = r * 16.0;
    let i = min(i32(x), 15);
    return mix(c_Glow[i], c_Glow[i + 1], x - f32(i));
}

@fragment
fn main_ps(in: BallVertex) -> @location(0) vec4<f32> {
    // Heat saturates here, not where it is added up: a few hard hits still read as hotter than one.
    // The color goes from its own through red-orange and yellow to white, and the ball brightens.
    let glow = 1.0 - exp(-in.heat);
    var color = SampleGradient(in.gradientPos);
    color = mix(color, vec3<f32>(1.0, 0.25, 0.04), saturate(glow * 3.0));
    color = mix(color, vec3<f32>(1.0, 0.85, 0.5), saturate(glow * 3.0 - 1.0));
    color = mix(color, vec3<f32>(1.0, 1.0, 1.0), saturate(glow * 3.0 - 2.0));

    let r = length(in.spriteCoord);
    // Inside the radius: a ball, brighter towards its center. Outside: a faint glow.
    let ballR = r * SPRITE_EXTENT;

    // A lit ball: its surface normal in view space (z away from the camera), a light from the upper
    // left in front, and a highlight (white, brighter on the cool ones: a hot ball glows over it).
    var brightness = 0.5 * SampleGlow(r);
    var highlight = 0.0;
    var mark = 0.0;
    if (ballR < 1.0) {
        let normal = vec3<f32>(in.spriteCoord * SPRITE_EXTENT, -sqrt(1.0 - ballR * ballR));
        if (in.lit > 0.5) {
            let toLight = normalize(vec3<f32>(-0.45, 0.55, -0.7));
            brightness = 0.2 + 0.9 * saturate(dot(normal, toLight));
            // Half way between the light and the viewer (at -z).
            highlight = 0.5 * pow(saturate(dot(normal, normalize(toLight + vec3<f32>(0.0, 0.0, -1.0)))), 24.0)
                * (1.0 - saturate(glow * 3.0));
        } else {
            // Unlit: brighter towards the center, as the balls were first drawn.
            brightness = 0.35 + 0.65 * sqrt(1.0 - ballR * ballR);
        }
        // A dark mark at the two poles of the particle's marker direction, so a spinning ball shows it.
        mark = smoothstep(0.8, 0.9, abs(dot(normal, in.marker)));
    }
    return vec4<f32>(brightness * (1.0 - 0.6 * mark) * (1.0 + 0.3 * glow) * color + vec3<f32>(highlight), 1.0);
}

// --- Axis gizmo ----------------------------------------------------------------------------

// Half the width of a segment, in pixels.
const GIZMO_HALF_WIDTH = 1.5;

struct ColorVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn axis_vs(@builtin(vertex_index) vertexIndex: u32) -> ColorVertex {
    var out: ColorVertex;
    let segment = vertexIndex / 6u;
    let halfViewport = 0.5 / g_Gizmo.header.xy;

    // In pixels from the center of the viewport, y up.
    let points = g_Gizmo.points[segment];
    let p0 = vec2<f32>(points.x - halfViewport.x, halfViewport.y - points.y);
    let p1 = vec2<f32>(points.z - halfViewport.x, halfViewport.y - points.w);

    var along = p1 - p0;
    let segmentLength = length(along);
    along = select(vec2<f32>(1.0, 0.0), along / max(segmentLength, 1e-6), segmentLength > 1e-4);
    // The quad goes past the ends by the half width.
    let across = vec2<f32>(-along.y, along.x) * GIZMO_HALF_WIDTH;
    along = along * GIZMO_HALF_WIDTH;

    var corners = array<vec2<f32>, 4>(p0 - along - across, p0 - along + across, p1 + along + across, p1 + along - across);
    let position = corners[c_GizmoQuadIndex[vertexIndex % 6u]];

    out.position = vec4<f32>(position * g_Gizmo.header.xy * 2.0, 0.0, 1.0);
    out.color = g_Gizmo.colors[segment];
    return out;
}

@fragment
fn color_ps(in: ColorVertex) -> @location(0) vec4<f32> {
    return in.color;
}

// --- Coordinate planes ---------------------------------------------------------------------

const PLANE_HALF_LINES = 6;
const PLANE_LINES_PER_DIRECTION = 13u;
const PLANE_LINES_PER_PLANE = 26u;
// World distance between lines, and half the width of a line in pixels.
const PLANE_SPACING = 2.0;
const PLANE_HALF_WIDTH = 0.75;
const PLANE_NEAR = 0.1;

@vertex
fn plane_vs(@builtin(vertex_index) vertexIndex: u32) -> ColorVertex {
    var out: ColorVertex;
    out.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
    out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let lineIndex = vertexIndex / 6u;
    let plane = lineIndex / PLANE_LINES_PER_PLANE;
    let inPlane = lineIndex % PLANE_LINES_PER_PLANE;
    let direction = inPlane / PLANE_LINES_PER_DIRECTION;
    let index = i32(inPlane % PLANE_LINES_PER_DIRECTION) - PLANE_HALF_LINES;

    // The line at offset u along one in-plane axis, spanning the grid along the other.
    let u = f32(index) * PLANE_SPACING;
    let extent = f32(PLANE_HALF_LINES) * PLANE_SPACING;
    var a = vec2<f32>(-extent, u);
    var b = vec2<f32>(extent, u);
    if (direction == 0u) {
        a = vec2<f32>(u, -extent);
        b = vec2<f32>(u, extent);
    }

    // Plane 0 is XY, 1 is XZ, 2 is YZ.
    var p0 = vec3<f32>(0.0, a.x, a.y);
    var p1 = vec3<f32>(0.0, b.x, b.y);
    if (plane == 0u) {
        p0 = vec3<f32>(a.x, a.y, 0.0);
        p1 = vec3<f32>(b.x, b.y, 0.0);
    } else if (plane == 1u) {
        p0 = vec3<f32>(a.x, 0.0, a.y);
        p1 = vec3<f32>(b.x, 0.0, b.y);
    }

    // To view space, where the depth is z: cut the line at the near plane (a grid reaching behind
    // the camera would otherwise flip across the screen).
    var e0 = g_View.worldToView * vec4<f32>(p0, 1.0);
    var e1 = g_View.worldToView * vec4<f32>(p1, 1.0);
    if (e0.z < PLANE_NEAR && e1.z < PLANE_NEAR) {
        return out;
    }
    if (e0.z < PLANE_NEAR) {
        e0 = mix(e0, e1, (PLANE_NEAR - e0.z) / (e1.z - e0.z));
    }
    if (e1.z < PLANE_NEAR) {
        e1 = mix(e1, e0, (PLANE_NEAR - e1.z) / (e0.z - e1.z));
    }

    let c0 = g_View.viewToClip * e0;
    let c1 = g_View.viewToClip * e1;

    // In pixels from the center of the viewport, y up, then as the gizmo's segments.
    let halfViewport = 0.5 / g_View.viewportSizeInv;
    let s0 = c0.xy / c0.w * halfViewport;
    let s1 = c1.xy / c1.w * halfViewport;
    var along = s1 - s0;
    let lineLength = length(along);
    along = select(vec2<f32>(1.0, 0.0), along / max(lineLength, 1e-6), lineLength > 1e-4);
    let across = vec2<f32>(-along.y, along.x) * PLANE_HALF_WIDTH;
    along = along * PLANE_HALF_WIDTH;

    var corners = array<vec2<f32>, 4>(s0 - along - across, s0 - along + across, s1 + along + across, s1 + along - across);
    let position = corners[c_GizmoQuadIndex[vertexIndex % 6u]];
    out.position = vec4<f32>(position * g_View.viewportSizeInv * 2.0, 0.0, 1.0);

    // The outermost lines and the ones through the origin (the axes) stand out.
    let emphasized = abs(index) == PLANE_HALF_LINES || index == 0;
    var tint = vec3<f32>(1.0, 0.3, 0.3);
    if (plane == 0u) {
        tint = vec3<f32>(0.3, 0.3, 1.0);
    } else if (plane == 1u) {
        tint = vec3<f32>(0.3, 1.0, 0.3);
    }
    out.color = vec4<f32>(tint, select(0.15, 0.45, emphasized));
    return out;
}

// --- Trails --------------------------------------------------------------------------------

// A streak behind each particle, along its velocity: a quad of 6 vertices from the head (the
// particle, as wide as its ball on screen) to the tail, narrowing and fading out, in the color of
// the particle's gradient position.
@vertex
fn trail_vs(@builtin(vertex_index) vertexIndex: u32) -> ColorVertex {
    var out: ColorVertex;
    out.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
    out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let particle = t_Particles[vertexIndex / 6u];
    let e0 = g_View.worldToView * vec4<f32>(particle.pos.xyz, 1.0);
    let e1 = g_View.worldToView * vec4<f32>(particle.pos.xyz - particle.vel.xyz * g_Trail.x, 1.0);
    // Not drawn when either end is behind the camera, or the particle has merged away.
    if (e0.z < PLANE_NEAR || e1.z < PLANE_NEAR || particle.info.x < 0.0) {
        return out;
    }

    let c0 = g_View.viewToClip * e0;
    let c1 = g_View.viewToClip * e1;
    let halfViewport = 0.5 / g_View.viewportSizeInv;
    let s0 = c0.xy / c0.w * halfViewport;
    let s1 = c1.xy / c1.w * halfViewport;

    var along = s1 - s0;
    let trailLength = length(along);
    // Shorter than a pixel: nothing to see.
    if (trailLength < 1.0) {
        return out;
    }
    along = along / trailLength;

    // The head as wide as the ball (its true radius over the depth, in pixels), at least a pixel.
    let radiusPixels = particle.marker.w * g_View.viewToClip[1][1] / e0.z * halfViewport.y;
    let across = vec2<f32>(-along.y, along.x) * clamp(0.8 * radiusPixels, 0.75, 8.0);

    var corners = array<vec2<f32>, 4>(s0 - across, s0 + across, s1 + across * 0.1, s1 - across * 0.1);
    let corner = c_GizmoQuadIndex[vertexIndex % 6u];
    out.position = vec4<f32>(corners[corner] * g_View.viewportSizeInv * 2.0, 0.0, 1.0);
    // Corners 0 and 1 are the head, 2 and 3 the tail.
    out.color = vec4<f32>(SampleGradient(particle.vel.w), select(0.0, g_Trail.y, corner < 2u));
    return out;
}
`;

// The bloom of compute_nbody_collisions: async_compute_bloom_post.hlsl's threshold and blur passes
// (Vulkan-Samples' async_compute), with the threshold and strength in a uniform buffer.
const POST_WGSL = `
struct BlurConstants {
    resolution: vec2<u32>,
    invResolution: vec2<f32>,
    invInputResolution: vec2<f32>,
    // The brightness above which the scene blooms, and how much of the excess is kept.
    threshold: f32,
    strength: f32,
};

@group(0) @binding(0) var t_Input: texture_2d<f32>;
@group(0) @binding(1) var s_Input: sampler;
@group(0) @binding(2) var u_Output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> g_Blur: BlurConstants;

fn getUV(uv: vec2<f32>, x: f32, y: f32, scale: f32) -> vec2<f32> {
    return uv + g_Blur.invInputResolution * (vec2<f32>(x, y) * scale);
}

fn sampleInput(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(t_Input, s_Input, uv, 0.0).rgb;
}

// blur_common.h
fn bloomBlur(uv: vec2<f32>, uvScale: f32) -> vec3<f32> {
    var rgb = vec3<f32>(0.0);
    let N = -1.0;
    let Z = 0.0;
    let P = 1.0;
    rgb += 0.25 * sampleInput(getUV(uv, Z, Z, uvScale));
    rgb += 0.0625 * sampleInput(getUV(uv, N, P, uvScale));
    rgb += 0.0625 * sampleInput(getUV(uv, P, P, uvScale));
    rgb += 0.0625 * sampleInput(getUV(uv, N, N, uvScale));
    rgb += 0.0625 * sampleInput(getUV(uv, P, N, uvScale));
    rgb += 0.125 * sampleInput(getUV(uv, N, Z, uvScale));
    rgb += 0.125 * sampleInput(getUV(uv, P, Z, uvScale));
    rgb += 0.125 * sampleInput(getUV(uv, Z, N, uvScale));
    rgb += 0.125 * sampleInput(getUV(uv, Z, P, uvScale));
    return rgb;
}

// threshold.comp
@compute @workgroup_size(8, 8, 1)
fn threshold_cs(@builtin(global_invocation_id) globalId: vec3<u32>) {
    if (all(globalId.xy < g_Blur.resolution)) {
        let uv = (vec2<f32>(globalId.xy) + 0.5) * g_Blur.invResolution;
        var rgb = sampleInput(uv);
        rgb = g_Blur.strength * max(rgb - vec3<f32>(g_Blur.threshold), vec3<f32>(0.0));
        textureStore(u_Output, vec2<i32>(globalId.xy), vec4<f32>(rgb, 1.0));
    }
}

// blur_down.comp: downscale pass.
@compute @workgroup_size(8, 8, 1)
fn blur_down_cs(@builtin(global_invocation_id) globalId: vec3<u32>) {
    if (all(globalId.xy < g_Blur.resolution)) {
        let uv = (vec2<f32>(globalId.xy) + 0.5) * g_Blur.invResolution;
        textureStore(u_Output, vec2<i32>(globalId.xy), vec4<f32>(bloomBlur(uv, 1.75), 1.0));
    }
}

// blur_up.comp: upscale pass.
@compute @workgroup_size(8, 8, 1)
fn blur_up_cs(@builtin(global_invocation_id) globalId: vec3<u32>) {
    if (all(globalId.xy < g_Blur.resolution)) {
        let uv = (vec2<f32>(globalId.xy) + 0.5) * g_Blur.invResolution;
        textureStore(u_Output, vec2<i32>(globalId.xy), vec4<f32>(bloomBlur(uv, 0.875), 1.0));
    }
}
`;

// The HDR frame and its bloom into the back buffer: a module of its own, its bind group is group 0.
const COMPOSITE_WGSL = `
// --- Composite: the HDR frame and its bloom into the back buffer ---------------------------

@group(0) @binding(0) var t_Hdr: texture_2d<f32>;
@group(0) @binding(1) var t_Bloom: texture_2d<f32>;
@group(0) @binding(2) var s_Linear: sampler;

struct CompositeVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// A triangle over the screen.
@vertex
fn composite_vs(@builtin(vertex_index) vertexId: u32) -> CompositeVertex {
    var out: CompositeVertex;
    var pos = vec2<f32>(-1.0, 3.0);
    if (vertexId == 0u) {
        pos = vec2<f32>(-1.0, -1.0);
    } else if (vertexId == 1u) {
        pos = vec2<f32>(3.0, -1.0);
    }
    out.uv = pos * 0.5 + 0.5;
    out.position = vec4<f32>(pos.x, -pos.y, 0.0, 1.0);
    return out;
}

// The particles add up above 1 (a pile of them, a hot one with its glow): a soft shoulder from
// KNEE on, per channel, so the colors below it are as they were and the rest saturates smoothly.
const KNEE = 0.8;

@fragment
fn composite_ps(in: CompositeVertex) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(t_Hdr, s_Linear, in.uv, 0.0).rgb + textureSampleLevel(t_Bloom, s_Linear, in.uv, 0.0).rgb;
    let over = max(color - vec3<f32>(KNEE), vec3<f32>(0.0));
    return vec4<f32>(min(color, vec3<f32>(KNEE)) + (1.0 - KNEE) * (vec3<f32>(1.0) - exp(-over / (1.0 - KNEE))), 1.0);
}
`;
