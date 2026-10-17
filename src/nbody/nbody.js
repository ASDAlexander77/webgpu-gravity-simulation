// compute_nbody_collisions (native_ts_graphics) on WebGPU: heavy centers with clouds of light
// particles around them, spheres with spin that bounce, roll, heat up and glow, merge or break.
// Ported from examples/compute_nbody_collisions.ts; the passes are in sim_shader.js and
// draw_shader.js, the controls in ui.js.

// The defaults of the attractors and the particles per attractor.
const NUM_ATTRACTORS = 6;
const PARTICLES_PER_ATTRACTOR = 1024;
const MAX_ATTRACTORS = 24;

// Seconds without input before the camera starts to orbit the center by itself, the seconds it
// takes to reach full speed, and that speed in radians per second (a turn in about 70 s).
const IDLE_SECONDS = 5.0;
const IDLE_RAMP_SECONDS = 2.0;
const IDLE_ROTATION_SPEED = 0.09;

// The compute passes run 256 threads per group.
const WORKGROUP_SIZE = 256;
// struct Particle: vec4 pos, vel, info, spin, marker.
const PARTICLE_FLOATS = 20;
// struct Contact: vec4 velocity, shift, spin, fragment.
const CONTACT_FLOATS = 16;
// struct SimulationConstants: f32 deltaT, i32 particleCount, i32 collisions, i32 heating, then the
// UI's values (the indices below).
const SIMULATION_CONSTANTS_SIZE = 80;
const SIMULATION_FLOATS = SIMULATION_CONSTANTS_SIZE / 4;
const CONSTANT_DELTA_T = 0;
const CONSTANT_PARTICLE_COUNT = 1;
const CONSTANT_COLLISIONS = 2;
const CONSTANT_HEATING = 3;
const CONSTANT_TIME_FACTOR = 4;
const CONSTANT_GRAVITY = 5;
const CONSTANT_RESTITUTION = 6;
const CONSTANT_HEAT_FACTOR = 7;
const CONSTANT_COOL_SECONDS = 8;
const CONSTANT_FRICTION = 9;
const CONSTANT_MERGE_SPEED = 10;
const CONSTANT_FULL_GRAVITY = 11;
const CONSTANT_FRAGMENT_SPEED = 12;
const CONSTANT_CONTACT_SPRINGS = 13;
const CONSTANT_RESTITUTION_SPEED = 14;
const CONSTANT_ROLLING_RESISTANCE = 15;
const CONSTANT_SIZE_SOFTENING = 16;
const CONSTANT_HOT_MERGE_SPEED = 17;

// The simulation's defaults, and what the UI's sliders can set them to.
const DEFAULT_TIME_FACTOR = 0.05;
const DEFAULT_GRAVITY = 0.002;
const DEFAULT_RESTITUTION = 0.8;
const DEFAULT_HEAT_FACTOR = 0.004;
const DEFAULT_COOL_SECONDS = 3.0;
const DEFAULT_FRICTION = 0.3;
// Bounciness falls as (this / impact speed)^(1/4) above this impact speed.
const DEFAULT_RESTITUTION_SPEED = 5.0;
const RESTITUTION_SPEED_MAX = 50.0;
const DEFAULT_ROLLING_RESISTANCE = 0.05;
const ROLLING_RESISTANCE_MAX = 0.5;
// Merging: particles touching slower than this (relative speed) merge; the switch is off, the
// slider at DEFAULT_MERGE_SLIDER.
const DEFAULT_MERGE_SPEED = 0.0;
const DEFAULT_MERGE_SLIDER = 2.0;
const MERGE_SPEED_MAX = 8.0;
// With merging: glowing particles merge up to this much faster.
const DEFAULT_HOT_MERGE_SPEED = 4.0;
const HOT_MERGE_SPEED_MAX = 20.0;
// Breaking: particles hit harder than this (approach speed) break in two, into free slots.
const DEFAULT_FRAGMENT_SPEED = 0.0;
const DEFAULT_FRAGMENT_SLIDER = 10.0;
const FRAGMENT_SPEED_MAX = 50.0;
const DEFAULT_FRAGMENT_ROOM = 0.0;
// -nocool's cooling time: the heat stays.
const NO_COOLING_SECONDS = 1.0e9;
// With merging on, the particles left are counted back from the GPU every this many frames.
const ALIVE_COUNT_FRAMES = 30;
// The sigma of a group's cloud around its attractor, and the random seed (the Park-Miller
// generator wants one from 1 up).
const DEFAULT_SPREAD = 0.75;
const DEFAULT_SEED = 12345;
// Collision radii: the heavy centers are balls the light particles bounce off, the light ones
// differ in size, and the mass follows the volume (20 to 312).
const HEAVY_RADIUS = 0.5;
const LIGHT_RADIUS_MIN = 0.04;
const LIGHT_RADIUS_MAX = 0.1;
const LIGHT_MASS_MIN = 20.0;
const HEAVY_MASS = 90000.0;
const HEAVY_RADIUS_MAX = 2.0;
const LIGHT_RADIUS_SLIDER_MIN = 0.01;
const LIGHT_RADIUS_SLIDER_MAX = 0.4;
// How the light particles are placed around their attractor, and how the attractors are laid out.
const PARTICLE_LAYOUTS = ['Gaussian cloud', 'Uniform ball', 'Spherical shell', 'Flat disk'];
const PARTICLE_LAYOUT_GAUSSIAN = 0;
const PARTICLE_LAYOUT_BALL = 1;
const PARTICLE_LAYOUT_SHELL = 2;
const PARTICLE_LAYOUT_DISK = 3;
const ATTRACTOR_LAYOUTS = ['Preset (random beyond 6)', 'Random in a ball', 'Ring', 'Sphere', 'Line'];
const ATTRACTOR_LAYOUT_PRESET = 0;
const ATTRACTOR_LAYOUT_RANDOM = 1;
const ATTRACTOR_LAYOUT_RING = 2;
const ATTRACTOR_LAYOUT_SPHERE = 3;
const ATTRACTOR_LAYOUT_LINE = 4;
const DEFAULT_LAYOUT_RADIUS = 8.0;
const LAYOUT_RADIUS_MAX = 30.0;
// The UI's choices of particles per attractor: the gravity pass expects the heavy centers at
// multiples of 1024 (but any count works up to 256 particles in all, as with 1).
const PARTICLE_CHOICES = [1, 1024, 2048, 4096, 8192];
// Longest simulation step, in seconds: a stall would otherwise move the particles through each
// other in one step.
const MAX_DELTA_T = 1.0 / 30.0;
// Positions drawn for a light particle before it is kept overlapping.
const MAX_PLACEMENT_ATTEMPTS = 100;
// The quad each particle is drawn as.
const VERTICES_PER_PARTICLE = 6;
// The axis gizmo: up to GIZMO_MAX_SEGMENTS segments of 6 vertices, in a uniform buffer of a vec4
// (1 / viewport size), then every segment's endpoints and color. Sizes are for a 720 pixel high
// window and scale with it.
const GIZMO_MAX_SEGMENTS = 16;
const GIZMO_FLOATS = 4 + GIZMO_MAX_SEGMENTS * 8;
const GIZMO_ORIGIN = 70.0;
const GIZMO_LENGTH = 45.0;
const GIZMO_LABEL_DISTANCE = 15.0;
const GIZMO_LABEL_HALF_WIDTH = 5.0;
const GIZMO_LABEL_HALF_HEIGHT = 7.0;
// The coordinate planes: 3 planes of 2 * 13 grid lines, 6 vertices each.
const PLANE_LINE_COUNT = 3 * 2 * 13;
// The view constants: two matrices, then 1 / viewport size (padded to a vec4).
const VIEW_FLOATS = 36;
// The bloom: the scene is drawn into an RGBA16F target, a threshold pass keeps what is brighter
// than the threshold (times the strength) at half size, BLUR_LEVELS levels blur it down and back up,
// and the composite adds level BLOOM_INDEX to the scene.
const BLUR_FLOATS = 8;
const BLUR_LEVELS = 6;
const BLOOM_INDEX = 1;
const BLUR_GROUP_SIZE = 8;
const DEFAULT_BLOOM_THRESHOLD = 1.0;
const DEFAULT_BLOOM_STRENGTH = 0.3;
const TRAIL_SECONDS_MAX = 3.0;
const DEFAULT_TRAIL_SECONDS = 0.35;
const TRAIL_OPACITY = 0.25;
const HDR_FORMAT = 'rgba16float';

// The size of a level of the blur chain: level 0 is the target's.
function levelSize(size, level) {
    return Math.max(1, size >> level);
}

function radians(degrees) {
    return degrees * Math.PI / 180.0;
}

// --- Random numbers ------------------------------------------------------------------------

// std::default_random_engine replacement: a Park-Miller generator, so that a seed gives the same
// simulation every time.
let g_RandomSeed = DEFAULT_SEED;
function randomFloat() {
    g_RandomSeed = (g_RandomSeed * 16807.0) % 2147483647.0;
    return g_RandomSeed / 2147483647.0;
}

// std::normal_distribution(0, 1) replacement: Box-Muller over randomFloat, which never returns 0.
function randomNormal() {
    const u1 = randomFloat();
    const u2 = randomFloat();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

// math::perspProjD3DStyleReverse(verticalFOV, aspect, zNear): reverse Z, infinite far plane,
// row-major, for row vectors.
function perspProjD3DStyleReverse(verticalFOV, aspect, zNear) {
    const yScale = 1.0 / Math.tan(0.5 * verticalFOV);
    const xScale = yScale / aspect;
    return [
        xScale, 0.0, 0.0, 0.0,
        0.0, yScale, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, zNear, 0.0,
    ];
}

function cross3(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalize3(a) {
    const length = Math.hypot(a[0], a[1], a[2]) || 1.0;
    return [a[0] / length, a[1] / length, a[2] / length];
}

// --- The camera ----------------------------------------------------------------------------

// A look-at camera orbiting its target: compute_nbody's, 14 units from the origin, rotated (-26,
// 75) degrees. View space has x to the right, y up and z away from the camera.
class OrbitCamera {
    constructor() {
        this.target = [0.0, 0.0, 0.0];
        this.distance = 14.0;
        this.yaw = radians(75.0);
        this.pitch = radians(26.0);
    }

    rotate(deltaYaw, deltaPitch) {
        this.yaw += deltaYaw;
        const limit = radians(89.0);
        this.pitch = Math.max(-limit, Math.min(limit, this.pitch + deltaPitch));
    }

    zoom(factor) {
        this.distance = Math.max(1.5, Math.min(120.0, this.distance * factor));
    }

    // The world to view matrix, row-major for row vectors (Donut's): the rows are the world axes
    // in view space, the last one the translation.
    getWorldToView(out) {
        const cp = Math.cos(this.pitch);
        const position = [
            this.target[0] + this.distance * cp * Math.sin(this.yaw),
            this.target[1] + this.distance * Math.sin(this.pitch),
            this.target[2] + this.distance * cp * Math.cos(this.yaw),
        ];
        const forward = normalize3([this.target[0] - position[0], this.target[1] - position[1], this.target[2] - position[2]]);
        const right = normalize3(cross3([0.0, 1.0, 0.0], forward));
        const up = cross3(forward, right);
        const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        out[0] = right[0]; out[1] = up[0]; out[2] = forward[0]; out[3] = 0.0;
        out[4] = right[1]; out[5] = up[1]; out[6] = forward[1]; out[7] = 0.0;
        out[8] = right[2]; out[9] = up[2]; out[10] = forward[2]; out[11] = 0.0;
        out[12] = -dot(position, right); out[13] = -dot(position, up); out[14] = -dot(position, forward); out[15] = 1.0;
    }
}

// --- The simulation ------------------------------------------------------------------------

class NBodyCollisions {
    constructor(device, context, canvas, format, options = {}) {
        this.device = device;
        this.context = context;
        this.canvas = canvas;
        this.format = format;
        this.onStatus = null;

        this.camera = new OrbitCamera();
        this.idleSeconds = 0.0;
        this.buttonsDown = 0;
        this.autoRotate = true;

        this.paused = false;
        this.collisions = true;
        this.heating = true;
        this.fullGravity = false;
        this.showPlanes = true;
        this.showTrails = true;
        this.showGizmo = true;
        this.litSpheres = true;
        this.spinMarks = true;
        this.bloomOn = true;
        this.bloomThreshold = DEFAULT_BLOOM_THRESHOLD;
        this.bloomStrength = DEFAULT_BLOOM_STRENGTH;
        this.trailSeconds = DEFAULT_TRAIL_SECONDS;

        // The UI's switches: each feature on or off, and the sliders' values, which the simulation
        // uses only while their switch is on (applySwitches).
        this.frictionOn = true;
        this.frictionValue = DEFAULT_FRICTION;
        this.mergeOn = false;
        this.mergeValue = DEFAULT_MERGE_SLIDER;
        this.hotStickOn = true;
        this.hotStickValue = DEFAULT_HOT_MERGE_SPEED;
        this.breakOn = false;
        this.breakValue = DEFAULT_FRAGMENT_SLIDER;
        this.contactSprings = true;
        this.restitutionSpeedOn = true;
        this.restitutionSpeedValue = DEFAULT_RESTITUTION_SPEED;
        this.rollingOn = true;
        this.rollingValue = DEFAULT_ROLLING_RESISTANCE;
        this.sizeSoftening = true;

        // The UI's settings applied on Restart (the simulation's own are those of the running one).
        this.numAttractors = options.attractors ?? NUM_ATTRACTORS;
        this.particlesPerAttractor = options.particles ?? PARTICLES_PER_ATTRACTOR;
        this.fragmentRoom = options.room ?? DEFAULT_FRAGMENT_ROOM;
        this.spread = DEFAULT_SPREAD;
        this.ui = {
            attractors: this.numAttractors,
            particles: this.particlesPerAttractor,
            spread: DEFAULT_SPREAD,
            seed: options.seed ?? DEFAULT_SEED,
            heavyRadius: HEAVY_RADIUS,
            lightRadiusMin: LIGHT_RADIUS_MIN,
            lightRadiusMax: LIGHT_RADIUS_MAX,
            particleLayout: PARTICLE_LAYOUT_GAUSSIAN,
            attractorLayout: ATTRACTOR_LAYOUT_PRESET,
            layoutRadius: DEFAULT_LAYOUT_RADIUS,
            fragmentRoom: this.fragmentRoom,
        };
        this.restartRequested = false;
        this.numParticles = 0;
        this.placedCount = 0;
        this.aliveCount = 0;
        this.framesSinceCount = 0;
        this.readbackPending = false;
        this.statusSeconds = 0.0;
        this.error = '';

        // The SimulationConstants uniform, with its integer fields.
        this.constantsData = new ArrayBuffer(SIMULATION_CONSTANTS_SIZE);
        this.f32 = new Float32Array(this.constantsData);
        this.i32 = new Int32Array(this.constantsData);
        this.resetSimulationConstants();
        this.i32[CONSTANT_COLLISIONS] = 1;
        this.i32[CONSTANT_HEATING] = 1;
        if (options.noCool) {
            this.f32[CONSTANT_COOL_SECONDS] = NO_COOLING_SECONDS;
        }
        if (options.fullGravity) {
            this.setFullGravity(true);
        }
        if (options.merge > 0.0) {
            this.mergeOn = true;
            this.mergeValue = options.merge;
        }
        if (options.fragment > 0.0) {
            this.breakOn = true;
            this.breakValue = options.fragment;
        }
        this.applySwitches();

        this.viewConstants = new Float32Array(VIEW_FLOATS);
        this.viewMatrix = new Float32Array(16);
        this.projMatrix = new Float32Array(16);
        this.trailConstants = new Float32Array([0.0, TRAIL_OPACITY, 0.0, 0.0]);
        this.gizmoConstants = new Float32Array(GIZMO_FLOATS);
        this.gizmoSegments = 0;
        this.bloomWidth = 0;
        this.bloomHeight = 0;
        this.freeCounts = new Uint32Array([0, 0]);
    }

    // --- Settings ---------------------------------------------------------------------------

    setFullGravity(full) {
        this.fullGravity = full;
        this.i32[CONSTANT_FULL_GRAVITY] = full ? 1 : 0;
    }

    setCollisions(on) {
        this.collisions = on;
        this.i32[CONSTANT_COLLISIONS] = on ? 1 : 0;
    }

    setHeating(on) {
        this.heating = on;
        this.i32[CONSTANT_HEATING] = on ? 1 : 0;
    }

    // The switches into the simulation's constants: a feature switched off gets 0 (friction,
    // merging and breaking all stop at 0), on gets its slider's value.
    applySwitches() {
        this.f32[CONSTANT_FRICTION] = this.frictionOn ? this.frictionValue : 0.0;
        this.f32[CONSTANT_MERGE_SPEED] = this.mergeOn ? this.mergeValue : 0.0;
        this.f32[CONSTANT_HOT_MERGE_SPEED] = this.hotStickOn ? this.hotStickValue : 0.0;
        this.f32[CONSTANT_FRAGMENT_SPEED] = this.breakOn ? this.breakValue : 0.0;
        this.i32[CONSTANT_CONTACT_SPRINGS] = this.contactSprings ? 1 : 0;
        this.f32[CONSTANT_RESTITUTION_SPEED] = this.restitutionSpeedOn ? this.restitutionSpeedValue : 0.0;
        this.f32[CONSTANT_ROLLING_RESISTANCE] = this.rollingOn ? this.rollingValue : 0.0;
        this.i32[CONSTANT_SIZE_SOFTENING] = this.sizeSoftening ? 1 : 0;
    }

    // The simulation's live settings back to their defaults (the UI's Defaults button); the
    // switches stay as they are.
    resetSimulationConstants() {
        this.f32[CONSTANT_TIME_FACTOR] = DEFAULT_TIME_FACTOR;
        this.f32[CONSTANT_GRAVITY] = DEFAULT_GRAVITY;
        this.f32[CONSTANT_RESTITUTION] = DEFAULT_RESTITUTION;
        this.f32[CONSTANT_HEAT_FACTOR] = DEFAULT_HEAT_FACTOR;
        this.f32[CONSTANT_COOL_SECONDS] = DEFAULT_COOL_SECONDS;
        this.frictionValue = DEFAULT_FRICTION;
        this.restitutionSpeedValue = DEFAULT_RESTITUTION_SPEED;
        this.rollingValue = DEFAULT_ROLLING_RESISTANCE;
        this.mergeValue = DEFAULT_MERGE_SLIDER;
        this.hotStickValue = DEFAULT_HOT_MERGE_SPEED;
        this.breakValue = DEFAULT_FRAGMENT_SLIDER;
        this.applySwitches();
    }

    requestRestart() {
        this.restartRequested = true;
    }

    // --- Input ------------------------------------------------------------------------------

    onKey(code) {
        this.idleSeconds = 0.0;
        if (code === 'Space') {
            this.paused = !this.paused;
        } else if (code === 'KeyC') {
            this.setCollisions(!this.collisions);
        } else if (code === 'KeyH') {
            this.setHeating(!this.heating);
        } else if (code === 'KeyP') {
            this.showPlanes = !this.showPlanes;
        } else if (code === 'KeyT') {
            this.showTrails = !this.showTrails;
        } else {
            return false;
        }
        return true;
    }

    attachInput() {
        const canvas = this.canvas;
        let last = null;
        canvas.addEventListener('pointerdown', (e) => {
            this.idleSeconds = 0.0;
            this.buttonsDown++;
            last = [e.clientX, e.clientY];
            canvas.setPointerCapture(e.pointerId);
        });
        canvas.addEventListener('pointermove', (e) => {
            // The mouse moving over the window isn't input for the camera, dragging is.
            if (this.buttonsDown > 0 && last) {
                this.idleSeconds = 0.0;
                this.camera.rotate(-(e.clientX - last[0]) * 0.005, (e.clientY - last[1]) * 0.005);
                last = [e.clientX, e.clientY];
            }
        });
        const release = () => {
            this.idleSeconds = 0.0;
            if (this.buttonsDown > 0) {
                this.buttonsDown--;
            }
            last = null;
        };
        canvas.addEventListener('pointerup', release);
        canvas.addEventListener('pointercancel', release);
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.idleSeconds = 0.0;
            this.camera.zoom(Math.exp(e.deltaY * 0.001));
        }, { passive: false });
    }

    // --- Particles --------------------------------------------------------------------------

    // Whether a light particle at (x, y, z) would overlap one of the first `count` particles or
    // any attractor's heavy center (those of later groups aren't placed yet).
    overlaps(particles, count, heavies, x, y, z, radius) {
        for (let i = 0; i < count; i++) {
            const q = i * PARTICLE_FLOATS;
            const dx = x - particles[q + 0];
            const dy = y - particles[q + 1];
            const dz = z - particles[q + 2];
            const reach = radius + particles[q + 8];
            if (dx * dx + dy * dy + dz * dz < reach * reach) {
                return true;
            }
        }
        for (let i = 0; i < this.numAttractors; i++) {
            const dx = x - heavies[i * 3 + 0];
            const dy = y - heavies[i * 3 + 1];
            const dz = z - heavies[i * 3 + 2];
            const reach = radius + this.ui.heavyRadius;
            if (dx * dx + dy * dy + dz * dz < reach * reach) {
                return true;
            }
        }
        return false;
    }

    // Where the attractors' heavy centers go, per the UI's layout: three numbers each. The
    // preset's are compute_nbody's, whose clouds sit at 1/1.5 of them (the cloud centers, which
    // the other layouts put at the heavy centers). The random positions keep apart.
    layoutAttractors(clouds, heavies) {
        const preset = [
            5.0, 0.0, 0.0,
            -5.0, 0.0, 0.0,
            0.0, 0.0, 5.0,
            0.0, 0.0, -5.0,
            0.0, 4.0, 0.0,
            0.0, -8.0, 0.0,
        ];
        const layout = this.ui.attractorLayout;
        const radius = this.ui.layoutRadius;
        const n = this.numAttractors;
        const minApart = 2.0 * this.ui.heavyRadius + 1.0;
        for (let i = 0; i < n; i++) {
            let x = 0.0;
            let y = 0.0;
            let z = 0.0;
            if (layout === ATTRACTOR_LAYOUT_PRESET && i < 6) {
                x = preset[i * 3 + 0] * 1.5;
                y = preset[i * 3 + 1] * 1.5;
                z = preset[i * 3 + 2] * 1.5;
                clouds.push(preset[i * 3 + 0], preset[i * 3 + 1], preset[i * 3 + 2]);
            } else {
                if (layout === ATTRACTOR_LAYOUT_RING) {
                    const angle = 2.0 * Math.PI * i / n;
                    x = radius * Math.cos(angle);
                    z = radius * Math.sin(angle);
                } else if (layout === ATTRACTOR_LAYOUT_SPHERE) {
                    // A Fibonacci spiral over the sphere (one at the pole for a single attractor).
                    const sy = n > 1 ? 1.0 - 2.0 * (i + 0.5) / n : 0.0;
                    const ring = Math.sqrt(1.0 - sy * sy);
                    x = radius * ring * Math.cos(i * 2.399963);
                    y = radius * sy;
                    z = radius * ring * Math.sin(i * 2.399963);
                } else if (layout === ATTRACTOR_LAYOUT_LINE) {
                    x = n > 1 ? radius * (2.0 * i / (n - 1) - 1.0) : 0.0;
                } else {
                    // Random in a ball (also the preset's attractors beyond the sixth), drawn
                    // again while too close to an earlier one.
                    for (let attempt = 0; attempt < MAX_PLACEMENT_ATTEMPTS; attempt++) {
                        const gx = randomNormal();
                        const gy = randomNormal();
                        const gz = randomNormal();
                        const length = Math.max(Math.sqrt(gx * gx + gy * gy + gz * gz), 1.0e-6);
                        const reach = radius * Math.pow(randomFloat(), 1.0 / 3.0) / length;
                        x = gx * reach;
                        y = gy * reach;
                        z = gz * reach;
                        let apart = true;
                        for (let k = 0; k < i; k++) {
                            const dx = x - heavies[k * 3 + 0];
                            const dy = y - heavies[k * 3 + 1];
                            const dz = z - heavies[k * 3 + 2];
                            if (dx * dx + dy * dy + dz * dz < minApart * minApart) {
                                apart = false;
                            }
                        }
                        if (apart) {
                            break;
                        }
                    }
                }
                clouds.push(x, y, z);
            }
            heavies.push(x, y, z);
        }
    }

    // compute_nbody's particles around their attractors, with collision radii: the particles
    // placed, then the free slots left for fragments (a negative radius: gone).
    createParticles() {
        const clouds = [];
        const heavies = [];
        this.layoutAttractors(clouds, heavies);
        const heavyRadius = this.ui.heavyRadius;
        const lightMin = this.ui.lightRadiusMin;
        const lightMax = Math.max(this.ui.lightRadiusMax, lightMin);
        const layout = this.ui.particleLayout;
        const perAttractor = this.particlesPerAttractor;
        this.placedCount = this.numAttractors * perAttractor;
        const numParticles = this.placedCount + Math.floor(this.placedCount * this.fragmentRoom);
        const particles = new Float32Array(numParticles * PARTICLE_FLOATS);

        for (let i = 0; i < this.numAttractors; i++) {
            const ax = clouds[i * 3 + 0];
            const ay = clouds[i * 3 + 1];
            const az = clouds[i * 3 + 2];

            for (let j = 0; j < perAttractor; j++) {
                const p = (i * perAttractor + j) * PARTICLE_FLOATS;

                if (j === 0) {
                    // First particle in group as heavy center of gravity
                    particles[p + 0] = heavies[i * 3 + 0];
                    particles[p + 1] = heavies[i * 3 + 1];
                    particles[p + 2] = heavies[i * 3 + 2];
                    const heavyRatio = heavyRadius / HEAVY_RADIUS;
                    particles[p + 3] = HEAVY_MASS * heavyRatio * heavyRatio * heavyRatio;
                    particles[p + 8] = heavyRadius;
                } else {
                    // Positive masses, following the volume: a collision's impulse is shared in
                    // proportion to them.
                    const sizeRandom = randomFloat();
                    const radius = lightMin + (lightMax - lightMin) * sizeRandom * sizeRandom;
                    const sizeRatio = radius / LIGHT_RADIUS_MIN;
                    const mass = LIGHT_MASS_MIN * sizeRatio * sizeRatio * sizeRatio;

                    // Drawn again while it overlaps a particle placed before or a heavy center:
                    // the particles would otherwise bounce off each other all at once in the
                    // first frames.
                    let px = 0.0;
                    let py = 0.0;
                    let pz = 0.0;
                    for (let attempt = 0; attempt < MAX_PLACEMENT_ATTEMPTS; attempt++) {
                        if (layout === PARTICLE_LAYOUT_GAUSSIAN) {
                            px = ax + randomNormal() * this.spread;
                            py = ay + randomNormal() * this.spread;
                            pz = az + randomNormal() * this.spread;
                        } else if (layout === PARTICLE_LAYOUT_DISK) {
                            // In the XZ plane (thin), out from the heavy center's surface.
                            const angle = 2.0 * Math.PI * randomFloat();
                            const reach = heavyRadius + 2.0 * this.spread * Math.sqrt(randomFloat());
                            px = ax + reach * Math.cos(angle);
                            py = ay + randomNormal() * 0.05 * this.spread;
                            pz = az + reach * Math.sin(angle);
                        } else {
                            // A uniform ball, or a thin shell, of heavyRadius + 2 spread.
                            const gx = randomNormal();
                            const gy = randomNormal();
                            const gz = randomNormal();
                            const length = Math.max(Math.sqrt(gx * gx + gy * gy + gz * gz), 1.0e-6);
                            const outer = heavyRadius + 2.0 * this.spread;
                            const fill = layout === PARTICLE_LAYOUT_BALL ? Math.pow(randomFloat(), 1.0 / 3.0) : 1.0;
                            const reach = outer * fill / length;
                            px = ax + gx * reach;
                            py = ay + gy * reach;
                            pz = az + gz * reach;
                        }
                        if (!this.overlaps(particles, p / PARTICLE_FLOATS, heavies, px, py, pz, radius)) {
                            break;
                        }
                    }

                    // Velocity: cross(position - attractor, angular) plus some randomness.
                    const sign = i % 2 === 0 ? 1.0 : -1.0;
                    const angularX = 0.5 * sign;
                    const angularY = 1.5 * sign;
                    const angularZ = 0.5 * sign;
                    const ex = px - ax;
                    const ey = py - ay;
                    const ez = pz - az;
                    const vx = ey * angularZ - ez * angularY + randomNormal();
                    const vy = ez * angularX - ex * angularZ + randomNormal();
                    const vz = ex * angularY - ey * angularX + randomNormal() * 0.025;

                    particles[p + 0] = px;
                    particles[p + 1] = py;
                    particles[p + 2] = pz;
                    particles[p + 3] = mass;
                    particles[p + 4] = vx;
                    particles[p + 5] = vy;
                    particles[p + 6] = vz;
                    particles[p + 8] = radius;
                }

                // Color gradient offset
                particles[p + 7] = i / this.numAttractors;

                // No spin; the marker's direction from the particle's index, spread evenly over
                // the sphere (a Fibonacci spiral: not from randomFloat, which would move the
                // particles placed after it).
                const index = i * perAttractor + j;
                const markerZ = 1.0 - 2.0 * ((index % 997) + 0.5) / 997.0;
                const markerRing = Math.sqrt(1.0 - markerZ * markerZ);
                particles[p + 16] = markerRing * Math.cos(index * 2.399963);
                particles[p + 17] = markerRing * Math.sin(index * 2.399963);
                particles[p + 18] = markerZ;
                // A solid sphere's inertia over its mass (spin.w), and the true radius: the
                // particle's own, until it merges.
                particles[p + 15] = 0.4 * particles[p + 8] * particles[p + 8];
                particles[p + 19] = particles[p + 8];
            }
        }

        for (let i = this.placedCount; i < numParticles; i++) {
            particles[i * PARTICLE_FLOATS + 8] = -1.0;
            particles[i * PARTICLE_FLOATS + 16] = 1.0;
        }
        return particles;
    }

    // New particles in new buffers, from the UI's settings, and the bind groups over them. False
    // (the running simulation kept) when they would not fit in a buffer.
    createSimulation() {
        const device = this.device;
        this.numAttractors = this.ui.attractors;
        this.particlesPerAttractor = this.ui.particles;
        this.spread = this.ui.spread;
        this.fragmentRoom = this.ui.fragmentRoom;

        const placed = this.numAttractors * this.particlesPerAttractor;
        const slots = placed + Math.floor(placed * this.fragmentRoom);
        const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
        if (slots * PARTICLE_FLOATS * 4 > limit) {
            this.error = `${slots} particles do not fit in a GPU buffer (limit ${Math.floor(limit / 80)})`;
            return false;
        }
        this.error = '';

        g_RandomSeed = this.ui.seed;
        const particles = this.createParticles();
        const count = particles.length / PARTICLE_FLOATS;

        for (const buffer of [this.particleBuffer, this.contactBuffer, this.freeBuffer]) {
            if (buffer) {
                buffer.destroy();
            }
        }
        this.numParticles = count;
        this.particleBuffer = device.createBuffer({
            label: 'Particles',
            size: count * PARTICLE_FLOATS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        device.queue.writeBuffer(this.particleBuffer, 0, particles);
        // Written by collide_cs before integrate_cs reads it, every frame.
        this.contactBuffer = device.createBuffer({
            label: 'Contacts',
            size: count * CONTACT_FLOATS * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        // The free slots: 2 counts, then up to numParticles indices.
        this.freeBuffer = device.createBuffer({
            label: 'Free slots',
            size: (count + 2) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.i32[CONSTANT_PARTICLE_COUNT] = count;
        this.aliveCount = this.placedCount;
        this.framesSinceCount = 0;

        this.computeBindGroup = device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.simulationBuffer } },
                { binding: 1, resource: { buffer: this.particleBuffer } },
                { binding: 2, resource: { buffer: this.contactBuffer } },
                { binding: 3, resource: { buffer: this.freeBuffer } },
            ],
        });
        this.drawBindGroup = device.createBindGroup({
            layout: this.drawBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.viewBuffer } },
                { binding: 1, resource: { buffer: this.particleBuffer } },
                { binding: 2, resource: { buffer: this.trailBuffer } },
                { binding: 3, resource: { buffer: this.gizmoBuffer } },
            ],
        });
        return true;
    }

    // --- Setup ------------------------------------------------------------------------------

    // Compile the shaders, make the pipelines and the first simulation. Resolves to a list of
    // shader errors (empty if all is well).
    async init() {
        const device = this.device;
        const simModule = device.createShaderModule({ label: 'Simulation', code: SIM_WGSL });
        const drawModule = device.createShaderModule({ label: 'Draw', code: DRAW_WGSL });
        const postModule = device.createShaderModule({ label: 'Bloom', code: POST_WGSL });
        const compositeModule = device.createShaderModule({ label: 'Composite', code: COMPOSITE_WGSL });
        const errors = [];
        for (const module of [simModule, drawModule, postModule, compositeModule]) {
            const info = await module.getCompilationInfo();
            for (const message of info.messages) {
                if (message.type === 'error') {
                    errors.push(`${module.label}:${message.lineNum}:${message.linePos}: ${message.message}`);
                }
            }
        }
        if (errors.length > 0) {
            return errors;
        }

        const COMPUTE = GPUShaderStage.COMPUTE;
        const VERTEX = GPUShaderStage.VERTEX;
        const FRAGMENT = GPUShaderStage.FRAGMENT;

        // Compute: the constants, the particles, their contacts and the free slots.
        this.computeBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        const computeLayout = device.createPipelineLayout({ bindGroupLayouts: [this.computeBindGroupLayout] });
        this.simulationPipelines = {};
        for (const entryPoint of ['calculate_cs', 'count_cs', 'collide_cs', 'absorb_cs', 'kill_cs',
            'collect_free_cs', 'integrate_cs', 'fragment_cs']) {
            this.simulationPipelines[entryPoint] = device.createComputePipeline({
                label: entryPoint,
                layout: computeLayout,
                compute: { module: simModule, entryPoint },
            });
        }

        // Drawing: the view constants and the particles, the trails' constants, the gizmo.
        this.drawBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: VERTEX, buffer: { type: 'uniform' } },
                { binding: 3, visibility: VERTEX, buffer: { type: 'uniform' } },
            ],
        });
        const drawLayout = device.createPipelineLayout({ bindGroupLayouts: [this.drawBindGroupLayout] });
        const additive = {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        };
        const alphaBlend = {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        };
        const drawPipeline = (label, vertex, fragment, format, blend) => device.createRenderPipeline({
            label,
            layout: drawLayout,
            vertex: { module: drawModule, entryPoint: vertex },
            fragment: { module: drawModule, entryPoint: fragment, targets: [{ format, blend }] },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
        });
        this.ballPipeline = drawPipeline('Balls', 'main_vs', 'main_ps', HDR_FORMAT, additive);
        this.trailPipeline = drawPipeline('Trails', 'trail_vs', 'color_ps', HDR_FORMAT, alphaBlend);
        this.planePipeline = drawPipeline('Planes', 'plane_vs', 'color_ps', this.format, alphaBlend);
        this.gizmoPipeline = drawPipeline('Gizmo', 'axis_vs', 'color_ps', this.format, alphaBlend);

        this.linearClamp = device.createSampler({
            magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
        });

        // The composite reads the scene and the bloom.
        this.compositeBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 1, visibility: FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: FRAGMENT, sampler: { type: 'filtering' } },
            ],
        });
        this.compositePipeline = device.createRenderPipeline({
            label: 'Composite',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.compositeBindGroupLayout] }),
            vertex: { module: compositeModule, entryPoint: 'composite_vs' },
            fragment: { module: compositeModule, entryPoint: 'composite_ps', targets: [{ format: this.format }] },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
        });

        // The bloom's passes: each reads a texture through the sampler, writes another, and has
        // its resolutions in a uniform.
        this.blurBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: COMPUTE, texture: { sampleType: 'float' } },
                { binding: 1, visibility: COMPUTE, sampler: { type: 'filtering' } },
                { binding: 2, visibility: COMPUTE, storageTexture: { access: 'write-only', format: HDR_FORMAT } },
                { binding: 3, visibility: COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        const blurLayout = device.createPipelineLayout({ bindGroupLayouts: [this.blurBindGroupLayout] });
        const blurPipeline = (entryPoint) => device.createComputePipeline({
            label: entryPoint, layout: blurLayout, compute: { module: postModule, entryPoint },
        });
        this.thresholdPipeline = blurPipeline('threshold_cs');
        this.blurDownPipeline = blurPipeline('blur_down_cs');
        this.blurUpPipeline = blurPipeline('blur_up_cs');

        // Buffers written every frame.
        const uniform = (label, size) => device.createBuffer({
            label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.simulationBuffer = uniform('SimulationConstants', SIMULATION_CONSTANTS_SIZE);
        this.viewBuffer = uniform('ViewConstants', VIEW_FLOATS * 4);
        this.trailBuffer = uniform('Trail', 16);
        this.gizmoBuffer = uniform('Gizmo', GIZMO_FLOATS * 4);

        if (!this.createSimulation()) {
            return [this.error];
        }
        this.attachInput();
        return [];
    }

    // The HDR target, the blur chain and the bind groups over them, for a frame of width x
    // height: made on the first frame and again when the size changes.
    ensureBloomTargets(width, height) {
        if (width === this.bloomWidth && height === this.bloomHeight) {
            return;
        }
        const device = this.device;
        if (this.bloomWidth > 0) {
            this.hdrTexture.destroy();
            for (const texture of this.blurChain) {
                texture.destroy();
            }
            for (const buffer of this.blurBuffers) {
                buffer.destroy();
            }
        }
        this.bloomWidth = width;
        this.bloomHeight = height;

        this.hdrTexture = device.createTexture({
            label: 'HDR Color',
            size: [width, height],
            format: HDR_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.hdrView = this.hdrTexture.createView();
        this.blurChain = [];
        for (let level = 1; level <= BLUR_LEVELS; level++) {
            this.blurChain.push(device.createTexture({
                label: `Blur Chain ${level}`,
                size: [levelSize(width, level), levelSize(height, level)],
                format: HDR_FORMAT,
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
            }));
        }
        const views = this.blurChain.map(texture => texture.createView());

        // One pass per bind group, in the order they run: threshold, down, up. Each has a uniform
        // of its own (the resolutions differ), written every frame.
        this.blurBuffers = [];
        this.blurPasses = [];
        const w = width;
        const h = height;
        const addPass = (pipeline, input, output, outWidth, outHeight, inWidth, inHeight) => {
            const buffer = device.createBuffer({
                size: BLUR_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this.blurBuffers.push(buffer);
            const data = new ArrayBuffer(BLUR_FLOATS * 4);
            const u32 = new Uint32Array(data);
            const f32 = new Float32Array(data);
            u32[0] = outWidth;
            u32[1] = outHeight;
            f32[2] = 1.0 / outWidth;
            f32[3] = 1.0 / outHeight;
            f32[4] = 1.0 / inWidth;
            f32[5] = 1.0 / inHeight;
            this.blurPasses.push({
                pipeline, buffer, f32, data, outWidth, outHeight,
                bindGroup: device.createBindGroup({
                    layout: this.blurBindGroupLayout,
                    entries: [
                        { binding: 0, resource: input },
                        { binding: 1, resource: this.linearClamp },
                        { binding: 2, resource: output },
                        { binding: 3, resource: { buffer } },
                    ],
                }),
            });
        };
        addPass(this.thresholdPipeline, this.hdrView, views[0], levelSize(w, 1), levelSize(h, 1), w, h);
        for (let index = 1; index < BLUR_LEVELS; index++) {
            addPass(this.blurDownPipeline, views[index - 1], views[index],
                levelSize(w, index + 1), levelSize(h, index + 1), levelSize(w, index), levelSize(h, index));
        }
        for (let index = BLUR_LEVELS - 2; index >= 1; index--) {
            addPass(this.blurUpPipeline, views[index + 1], views[index],
                levelSize(w, index + 1), levelSize(h, index + 1), levelSize(w, index + 2), levelSize(h, index + 2));
        }

        this.compositeBindGroup = device.createBindGroup({
            layout: this.compositeBindGroupLayout,
            entries: [
                { binding: 0, resource: this.hdrView },
                { binding: 1, resource: views[BLOOM_INDEX] },
                { binding: 2, resource: this.linearClamp },
            ],
        });
    }

    // --- The gizmo --------------------------------------------------------------------------

    // Adds a segment from (x0, y0) to (x1, y1), in pixels from the top left, to the gizmo.
    addGizmoSegment(x0, y0, x1, y1, r, g, b, a) {
        if (this.gizmoSegments >= GIZMO_MAX_SEGMENTS) {
            return;
        }
        const points = 4 + this.gizmoSegments * 4;
        const colors = 4 + (GIZMO_MAX_SEGMENTS + this.gizmoSegments) * 4;
        this.gizmoConstants.set([x0, y0, x1, y1], points);
        this.gizmoConstants.set([r, g, b, a], colors);
        this.gizmoSegments++;
    }

    // The world X, Y and Z axes as the camera sees them: each axis direction in view space is a
    // row of the view matrix (row-vector convention), x to the right, y up and z away from the
    // camera. Axes pointing away are fainter, and each has its letter beyond the tip.
    buildGizmo(width, height) {
        const scale = height / 720.0;
        const originX = GIZMO_ORIGIN * scale;
        const originY = GIZMO_ORIGIN * scale;
        this.gizmoConstants[0] = 1.0 / width;
        this.gizmoConstants[1] = 1.0 / height;
        this.gizmoSegments = 0;

        for (let axis = 0; axis < 3; axis++) {
            const viewX = this.viewMatrix[axis * 4 + 0];
            const viewY = this.viewMatrix[axis * 4 + 1];
            const viewZ = this.viewMatrix[axis * 4 + 2];
            const dx = viewX * GIZMO_LENGTH * scale;
            const dy = -viewY * GIZMO_LENGTH * scale;
            const tipX = originX + dx;
            const tipY = originY + dy;
            const r = axis === 0 ? 1.0 : 0.3;
            const g = axis === 1 ? 1.0 : (axis === 0 ? 0.3 : 0.55);
            const b = axis === 2 ? 1.0 : 0.3;
            const a = 0.35 + 0.65 * (0.5 - 0.5 * viewZ);
            this.addGizmoSegment(originX, originY, tipX, tipY, r, g, b, a);

            // The letter, outward from the origin (above the tip when the axis points at the camera).
            const length = Math.sqrt(dx * dx + dy * dy);
            const cx = length > 1.0 ? tipX + dx / length * GIZMO_LABEL_DISTANCE * scale : tipX;
            const cy = length > 1.0 ? tipY + dy / length * GIZMO_LABEL_DISTANCE * scale : tipY - GIZMO_LABEL_DISTANCE * scale;
            const w = GIZMO_LABEL_HALF_WIDTH * scale;
            const h = GIZMO_LABEL_HALF_HEIGHT * scale;
            if (axis === 0) {
                this.addGizmoSegment(cx - w, cy - h, cx + w, cy + h, r, g, b, a);
                this.addGizmoSegment(cx - w, cy + h, cx + w, cy - h, r, g, b, a);
            } else if (axis === 1) {
                this.addGizmoSegment(cx - w, cy - h, cx, cy, r, g, b, a);
                this.addGizmoSegment(cx + w, cy - h, cx, cy, r, g, b, a);
                this.addGizmoSegment(cx, cy, cx, cy + h, r, g, b, a);
            } else {
                this.addGizmoSegment(cx - w, cy - h, cx + w, cy - h, r, g, b, a);
                this.addGizmoSegment(cx + w, cy - h, cx - w, cy + h, r, g, b, a);
                this.addGizmoSegment(cx - w, cy + h, cx + w, cy + h, r, g, b, a);
            }
        }
    }

    // --- Frames -----------------------------------------------------------------------------

    // Counts the particles left (the others merged into them), copied back from the GPU without
    // stalling the frame. Only ever fewer than at the start.
    startAliveCount(encoder) {
        if (this.readbackPending) {
            return;
        }
        this.readbackPending = true;
        const size = this.numParticles * PARTICLE_FLOATS * 4;
        const count = this.numParticles;
        const staging = this.device.createBuffer({
            size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        encoder.copyBufferToBuffer(this.particleBuffer, 0, staging, 0, size);
        this.pendingReadback = { staging, count };
    }

    async finishAliveCount() {
        const { staging, count } = this.pendingReadback;
        this.pendingReadback = null;
        try {
            await staging.mapAsync(GPUMapMode.READ);
            const particles = new Float32Array(staging.getMappedRange());
            let alive = 0;
            for (let i = 0; i < count; i++) {
                if (particles[i * PARTICLE_FLOATS + 8] >= 0.0) {
                    alive++;
                }
            }
            if (count === this.numParticles) {
                this.aliveCount = alive;
            }
            staging.unmap();
        } catch (e) {
            // The buffer went away with a restart: keep the last count.
        }
        staging.destroy();
        this.readbackPending = false;
    }

    // All the particles, read back (for checks from the console).
    async readParticles() {
        const size = this.numParticles * PARTICLE_FLOATS * 4;
        const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.particleBuffer, 0, staging, 0, size);
        this.device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const particles = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        return particles;
    }

    statusText() {
        if (this.error) {
            return this.error;
        }
        const collisionsInfo = (this.collisions ? 'collisions on' : 'collisions off')
            + (this.heating ? ', heating on' : ', heating off');
        return `${this.aliveCount} particles, ${collisionsInfo}${this.paused ? ' (paused)' : ''}`;
    }

    // Everything that changes with time, before this frame's commands exist.
    animate(elapsedSeconds) {
        if (this.restartRequested) {
            this.restartRequested = false;
            this.createSimulation();
        }

        // Left alone for a while, the camera drifts around the target, keeping its pitch and
        // distance, from wherever the user left it: speeding up so it doesn't start with a jerk.
        this.idleSeconds += elapsedSeconds;
        if (this.autoRotate && this.buttonsDown === 0 && this.idleSeconds > IDLE_SECONDS) {
            const ramp = Math.min((this.idleSeconds - IDLE_SECONDS) / IDLE_RAMP_SECONDS, 1.0);
            this.camera.rotate(IDLE_ROTATION_SPEED * ramp * elapsedSeconds, 0.0);
        }

        this.f32[CONSTANT_DELTA_T] = Math.min(elapsedSeconds, MAX_DELTA_T);

        // Merged-away particles don't show: count the ones left now and then.
        this.framesSinceCount++;
        this.countAliveNow = (this.f32[CONSTANT_MERGE_SPEED] > 0.0 || this.f32[CONSTANT_FRAGMENT_SPEED] > 0.0)
            && !this.paused && this.framesSinceCount >= ALIVE_COUNT_FRAMES && !this.readbackPending;

        this.statusSeconds += elapsedSeconds;
        if (this.statusSeconds > 0.25 && this.onStatus) {
            this.statusSeconds = 0.0;
            this.onStatus(this.statusText());
        }
    }

    render() {
        const device = this.device;
        const queue = device.queue;
        const canvas = this.canvas;
        const width = canvas.width;
        const height = canvas.height;
        this.ensureBloomTargets(width, height);

        // The camera: 60 degrees vertically, as in the sample.
        this.camera.getWorldToView(this.viewMatrix);
        this.projMatrix.set(perspProjD3DStyleReverse(radians(60.0), width / height, 0.1));
        this.viewConstants.set(this.viewMatrix, 0);
        this.viewConstants.set(this.projMatrix, 16);
        this.viewConstants[32] = 1.0 / width;
        this.viewConstants[33] = 1.0 / height;
        queue.writeBuffer(this.viewBuffer, 0, this.viewConstants);

        // The trails' constants and the balls' switches, read by both.
        this.trailConstants[0] = this.f32[CONSTANT_TIME_FACTOR] * this.trailSeconds;
        this.trailConstants[2] = this.litSpheres ? 1.0 : 0.0;
        this.trailConstants[3] = this.spinMarks ? 1.0 : 0.0;
        queue.writeBuffer(this.trailBuffer, 0, this.trailConstants);
        queue.writeBuffer(this.simulationBuffer, 0, this.constantsData);

        const encoder = device.createCommandEncoder();

        // Gravity, collisions, integration. Paused, nothing moves: the collision pass would still
        // push overlapping particles apart.
        if (!this.paused) {
            // Rounded up: the shaders skip the threads past particleCount.
            const groups = Math.ceil(this.numParticles / WORKGROUP_SIZE);
            const fragmenting = this.collisions && this.f32[CONSTANT_FRAGMENT_SPEED] > 0.0;
            if (fragmenting) {
                queue.writeBuffer(this.freeBuffer, 0, this.freeCounts);
            }
            // In one compute pass each dispatch sees the writes of the ones before.
            const pass = encoder.beginComputePass();
            pass.setBindGroup(0, this.computeBindGroup);
            const run = (name) => {
                pass.setPipeline(this.simulationPipelines[name]);
                pass.dispatchWorkgroups(groups);
            };
            run('calculate_cs');
            if (this.collisions) {
                run('count_cs');
                run('collide_cs');
                if (this.f32[CONSTANT_MERGE_SPEED] > 0.0) {
                    // The survivors absorb the merging particles, which are then removed.
                    run('absorb_cs');
                    run('kill_cs');
                }
            }
            if (fragmenting) {
                // The free slots (after the merges have freed theirs).
                run('collect_free_cs');
            }
            run('integrate_cs');
            if (fragmenting) {
                run('fragment_cs');
            }
            pass.end();
        }

        // The scene into the HDR target: the trails under the balls (they add up to a haze in the pile).
        const hdrPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.hdrView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store',
            }],
        });
        hdrPass.setBindGroup(0, this.drawBindGroup);
        if (this.showTrails && this.trailSeconds > 0.0) {
            hdrPass.setPipeline(this.trailPipeline);
            hdrPass.draw(this.numParticles * VERTICES_PER_PARTICLE);
        }
        hdrPass.setPipeline(this.ballPipeline);
        hdrPass.draw(this.numParticles * VERTICES_PER_PARTICLE);
        hdrPass.end();

        // The bloom: a threshold pass into the first blur level, then blurring down the levels and
        // back up to the second. With no strength, the chain stays black: the passes still run, as
        // the composite reads it.
        const strength = this.bloomOn ? this.bloomStrength : 0.0;
        const bloomPass = encoder.beginComputePass();
        for (const blur of this.blurPasses) {
            blur.f32[6] = this.bloomThreshold;
            blur.f32[7] = strength;
            queue.writeBuffer(blur.buffer, 0, blur.data);
            bloomPass.setPipeline(blur.pipeline);
            bloomPass.setBindGroup(0, blur.bindGroup);
            bloomPass.dispatchWorkgroups(Math.ceil(blur.outWidth / BLUR_GROUP_SIZE), Math.ceil(blur.outHeight / BLUR_GROUP_SIZE));
        }
        bloomPass.end();

        // The scene and the bloom into the back buffer, then the planes and the axes over them.
        const screenPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store',
            }],
        });
        screenPass.setPipeline(this.compositePipeline);
        screenPass.setBindGroup(0, this.compositeBindGroup);
        screenPass.draw(3);
        screenPass.setBindGroup(0, this.drawBindGroup);
        if (this.showPlanes) {
            screenPass.setPipeline(this.planePipeline);
            screenPass.draw(PLANE_LINE_COUNT * VERTICES_PER_PARTICLE);
        }
        if (this.showGizmo) {
            this.buildGizmo(width, height);
            queue.writeBuffer(this.gizmoBuffer, 0, this.gizmoConstants);
            screenPass.setPipeline(this.gizmoPipeline);
            screenPass.draw(this.gizmoSegments * VERTICES_PER_PARTICLE);
        }
        screenPass.end();

        if (this.countAliveNow) {
            this.countAliveNow = false;
            this.framesSinceCount = 0;
            this.startAliveCount(encoder);
        }
        queue.submit([encoder.finish()]);
        if (this.pendingReadback) {
            this.finishAliveCount();
        }
    }

    // One frame: the time since the last, then the commands.
    frame(now) {
        const elapsed = this.lastTime === undefined ? 0.0 : (now - this.lastTime) / 1000.0;
        this.lastTime = now;
        this.animate(elapsed);
        this.render();
    }
}
