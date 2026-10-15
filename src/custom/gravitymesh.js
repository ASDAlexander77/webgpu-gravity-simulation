
// compute_nbody_collisions (native_ts_graphics): heavy centers with clouds of light particles
// around them, spheres that bounce off each other instead of merging.
const GRAVITY_ATTRACTORS = [
    5.0, 0.0, 0.0,
    -5.0, 0.0, 0.0,
    0.0, 0.0, 5.0,
    0.0, 0.0, -5.0,
    0.0, 4.0, 0.0,
    0.0, -8.0, 0.0,
];
// Collision radii: the heavy centers are balls the light particles bounce off, the light ones
// grow with their mass (20 to 80).
const HEAVY_MASS = 90000.0;
const HEAVY_RADIUS = 0.5;
const LIGHT_RADIUS_BASE = 0.04;
const LIGHT_RADIUS_PER_MASS = 0.0005;
// Positions drawn for a light particle before it is kept overlapping.
const MAX_PLACEMENT_ATTEMPTS = 100;
// struct Particle in calc_wgsl: 4 x vec4<f32>.
const PARTICLE_FLOATS = 16;
// struct Contact in calc_wgsl: velocity change + flash, position correction.
const CONTACT_FLOATS = 8;
const WORKGROUP_SIZE = 64;
// The compute passes, in order, as the entry points of calc_wgsl.
const GRAVITY_PASSES = ['calculate_cs', 'count_cs', 'collide_cs', 'integrate_cs'];

function randomNormal() {
    // Box-Muller; 1 - random() is never 0.
    const u1 = 1.0 - Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

class GravityMesh extends ComputedMesh {
    // count: particles in all, spread over the attractors, the first of each group its heavy center.
    constructor(engine, count, numAttractors = 6) {
        super(engine);
        this.vertexCount = 4;

        numAttractors = Math.max(1, Math.min(numAttractors, GRAVITY_ATTRACTORS.length / 3));
        const particlesPerAttractor = Math.max(1, Math.floor(count / numAttractors));
        count = numAttractors * particlesPerAttractor;
        this.instanceCount = count;

        const instancesInitData = GravityMesh.#createParticles(numAttractors, particlesPerAttractor);

        const instanceLayout = [
            // info: x = color, y = collision flash, z = contacts this step
            { location: 0, type: 'float32', count: 4 },
            // position
            { location: 1, type: 'float32', count: 4 },
            // velocity
            { location: 2, type: 'float32', count: 4 },
            // mass, radius
            { location: 3, type: 'float32', count: 4 },
        ];

        this.instancesData = new InstanceData(
            engine,
            instancesInitData,
            instanceLayout
        );

        this.instancesSwapData = new InstanceData(
            engine,
            instancesInitData,
            instanceLayout
        );

        this.instancesCloneData = new InstanceCloneData(
            engine,
            instancesInitData,
            instanceLayout
        );

        // Written by collide_cs before integrate_cs reads it, every step.
        this.contactsData = new InstanceData(
            engine,
            new Array(count * CONTACT_FLOATS).fill(0),
            []
        );

        const paramsInitData = [
            // delta time, seconds (the shader scales it by TIME_FACTOR)
            1.0 / 60.0,
        ];

        this.paramsData = new UniformData(
            engine,
            paramsInitData,
        );

        const halfWidth = 0.5;
        this.vertexData = new VertexData(
            engine,
            [
                -halfWidth, halfWidth, 0.0, 0.0, 0.0, 0.0,
                -halfWidth, -halfWidth, 0.0, 0.0, 1.0, 0.0,
                halfWidth, halfWidth, 0.0, 0.0, 0.0, 1.0,
                halfWidth, -halfWidth, 0.0, 0.0, 1.0, 1.0
            ],
            [
                // x, y, z, 0
                { location: instanceLayout.length, type: 'float32', count: 4 },
                // UV
                { location: instanceLayout.length + 1, type: 'float32', count: 2 }
            ],
            'triangle-strip'
        );
    }

    // Several entry points instead of ComputedMesh's single 'main': kept in a field of its own so
    // that ComputedMesh doesn't dispatch it as well.
    set ComputeShader(elementId) {
        const computeShaderText = document.getElementById(elementId).value;
        this.passesShaderModule = this.engine.device.createShaderModule({ code: computeShaderText });
    }

    getPassesForComputeAndDrawCmd(commandEncoder, view, paramsData) {
        if (this.passesShaderModule) {
            // In one compute pass, each dispatch sees the writes of the ones before.
            const passEncoder = commandEncoder.beginComputePass();
            const bindGroup = this.tick % 2 ? this.#passesBindGroups()[0] : this.#passesBindGroups()[1];
            const workgroups = Math.ceil(this.instanceCount / WORKGROUP_SIZE);
            for (const pipeline of this.#passPipelines()) {
                passEncoder.setPipeline(pipeline);
                passEncoder.setBindGroup(0, bindGroup);
                passEncoder.setBindGroup(1, this.#paramsBindGroup());
                passEncoder.dispatchWorkgroups(workgroups);
            }
            passEncoder.end();
        }

        super.getPassesForComputeAndDrawCmd(commandEncoder, view, paramsData);
    }

    // One explicit layout for all the passes: bind groups made for a 'auto' layout only work with
    // the pipeline that made it.
    #passPipelineLayout() {
        if (this.passPipelineLayout) {
            return this.passPipelineLayout;
        }

        const device = this.engine.device;
        this.instancesBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        this.paramsBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        return this.passPipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [this.instancesBindGroupLayout, this.paramsBindGroupLayout],
        });
    }

    #passPipelines() {
        if (this.passPipelines) {
            return this.passPipelines;
        }

        const layout = this.#passPipelineLayout();
        return this.passPipelines = GRAVITY_PASSES.map(entryPoint =>
            this.engine.device.createComputePipeline({
                compute: { module: this.passesShaderModule, entryPoint },
                layout,
            }));
    }

    // [previous state in instancesData, previous state in instancesSwapData]
    #passesBindGroups() {
        if (this.passesBindGroups) {
            return this.passesBindGroups;
        }

        this.#passPipelineLayout();
        const create = (input, output) => this.engine.device.createBindGroup({
            layout: this.instancesBindGroupLayout,
            entries: [
                { binding: 0, resource: input.ResourceInfo },
                { binding: 1, resource: output.ResourceInfo },
                { binding: 2, resource: this.contactsData.ResourceInfo },
            ],
        });

        return this.passesBindGroups = [
            create(this.instancesData, this.instancesSwapData),
            create(this.instancesSwapData, this.instancesData),
        ];
    }

    #paramsBindGroup() {
        if (this.passesParamsBindGroup) {
            return this.passesParamsBindGroup;
        }

        this.#passPipelineLayout();
        return this.passesParamsBindGroup = this.engine.device.createBindGroup({
            layout: this.paramsBindGroupLayout,
            entries: [{ binding: 0, resource: this.paramsData.ResourceInfo }],
        });
    }

    // Whether a light particle at (x, y, z) would overlap one of the first `count` particles or
    // any attractor's heavy center (those of later groups aren't placed yet).
    static #overlaps(particles, count, numAttractors, x, y, z, radius) {
        for (let i = 0; i < count; i++) {
            const q = i * PARTICLE_FLOATS;
            const dx = x - particles[q + 4];
            const dy = y - particles[q + 5];
            const dz = z - particles[q + 6];
            const reach = radius + particles[q + 13];
            if (dx * dx + dy * dy + dz * dz < reach * reach) {
                return true;
            }
        }
        for (let i = 0; i < numAttractors; i++) {
            const dx = x - GRAVITY_ATTRACTORS[i * 3 + 0] * 1.5;
            const dy = y - GRAVITY_ATTRACTORS[i * 3 + 1] * 1.5;
            const dz = z - GRAVITY_ATTRACTORS[i * 3 + 2] * 1.5;
            const reach = radius + HEAVY_RADIUS;
            if (dx * dx + dy * dy + dz * dz < reach * reach) {
                return true;
            }
        }
        return false;
    }

    // compute_nbody's particles around their attractors, with collision radii.
    static #createParticles(numAttractors, particlesPerAttractor) {
        const numParticles = numAttractors * particlesPerAttractor;
        const particles = new Array(numParticles * PARTICLE_FLOATS).fill(0);

        for (let i = 0; i < numAttractors; i++) {
            const ax = GRAVITY_ATTRACTORS[i * 3 + 0];
            const ay = GRAVITY_ATTRACTORS[i * 3 + 1];
            const az = GRAVITY_ATTRACTORS[i * 3 + 2];

            for (let j = 0; j < particlesPerAttractor; j++) {
                const p = (i * particlesPerAttractor + j) * PARTICLE_FLOATS;

                if (j == 0) {
                    // First particle in group as heavy center of gravity
                    particles[p + 4] = ax * 1.5;
                    particles[p + 5] = ay * 1.5;
                    particles[p + 6] = az * 1.5;
                    particles[p + 12] = HEAVY_MASS;
                    particles[p + 13] = HEAVY_RADIUS;
                } else {
                    // A collision's impulse is shared in proportion to the masses.
                    const mass = 20.0 + 60.0 * Math.random();
                    const radius = LIGHT_RADIUS_BASE + LIGHT_RADIUS_PER_MASS * mass;

                    // Drawn again while it overlaps a particle placed before or a heavy center:
                    // overlapping particles would all be pushed apart in the first steps.
                    let px = 0.0;
                    let py = 0.0;
                    let pz = 0.0;
                    for (let attempt = 0; attempt < MAX_PLACEMENT_ATTEMPTS; attempt++) {
                        px = ax + randomNormal() * 0.75;
                        py = ay + randomNormal() * 0.75;
                        pz = az + randomNormal() * 0.75;
                        if (!GravityMesh.#overlaps(particles, p / PARTICLE_FLOATS, numAttractors, px, py, pz, radius)) {
                            break;
                        }
                    }

                    // Velocity: cross(position - attractor, angular) plus some randomness.
                    const sign = i % 2 == 0 ? 1.0 : -1.0;
                    const angularX = 0.5 * sign;
                    const angularY = 1.5 * sign;
                    const angularZ = 0.5 * sign;
                    const ex = px - ax;
                    const ey = py - ay;
                    const ez = pz - az;

                    particles[p + 4] = px;
                    particles[p + 5] = py;
                    particles[p + 6] = pz;
                    particles[p + 8] = ey * angularZ - ez * angularY + randomNormal();
                    particles[p + 9] = ez * angularX - ex * angularZ + randomNormal();
                    particles[p + 10] = ex * angularY - ey * angularX + randomNormal() * 0.025;
                    particles[p + 12] = mass;
                    particles[p + 13] = radius;
                }

                // Color: the group
                particles[p + 0] = i / numAttractors;
            }
        }

        return particles;
    }
}
