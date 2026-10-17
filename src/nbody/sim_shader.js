// compute_nbody_collisions (native_ts_graphics), the simulation passes in WGSL: the particles are
// spheres with spin that bounce off each other (and off the heavy centers) instead of passing
// through. Ported from shaders/compute_nbody_collisions.hlsl; the comments there explain the physics.
//
// Kept in a template string, not a <textarea>: a textarea decodes entities such as "&params".
const SIM_WGSL = `
struct Particle {
    pos: vec4<f32>,    // xyz = position, w = mass
    vel: vec4<f32>,    // xyz = velocity, w = gradient texture position
    info: vec4<f32>,   // x = collision radius (negative: merged into another particle, gone), y = heat,
                       // z = contacts this frame (count_cs), w = the radius it grows to this step
    spin: vec4<f32>,   // xyz = angular velocity, w = moment of inertia over the mass
    marker: vec4<f32>, // xyz = a direction fixed in the particle, w = its true radius
};

// What collide_cs found for a particle, applied by integrate_cs.
struct Contact {
    velocity: vec4<f32>, // xyz = velocity change, w = heat gained
    shift: vec4<f32>,    // xyz = position correction, w = 1 if the particle is merging with another
    spin: vec4<f32>,     // xyz = angular velocity change, w = the particle it merges into
    fragment: vec4<f32>, // w = 1 if the particle breaks in two; xyz = the halves' parting velocity
};

struct SimulationConstants {
    deltaT: f32,
    particleCount: i32,
    collisions: i32,
    heating: i32,
    timeFactor: f32,
    gravity: f32,
    restitution: f32,
    heatFactor: f32,
    coolSeconds: f32,
    friction: f32,
    mergeSpeed: f32,
    fullGravity: i32,
    fragmentSpeed: f32,
    contactSprings: i32,
    restitutionSpeed: f32,
    rollingResistance: f32,
    sizeSoftening: i32,
    hotMergeSpeed: f32,
    pad2: f32,
    pad3: f32,
};

// The free slots for the fragments: how many (collect_free_cs), how many are taken (fragment_cs),
// then their indices. The application zeroes the counts every step.
struct FreeSlots {
    count: atomic<u32>,
    taken: atomic<u32>,
    slots: array<u32>,
};

@group(0) @binding(0) var<uniform> sim: SimulationConstants;
@group(0) @binding(1) var<storage, read_write> u_Particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> u_Contacts: array<Contact>;
@group(0) @binding(3) var<storage, read_write> u_Free: FreeSlots;

const WORKGROUP_SIZE = 256u;
const SHARED_DATA_SIZE = 1024u;
const POWER = 0.75;
const SOFTEN = 0.05;
const FRAGMENT_MIN_RADIUS = 0.035;
const FRAGMENT_MAX_RADIUS = 0.3;
const CONTACT_STIFFNESS = 0.25;
const CONTACT_OVERLAP_MAX = 0.5;
const CONTACT_DAMPING = 1.0;
const GROWTH_PER_STEP = 0.003;
const SEPARATION = 0.5;

var<workgroup> s_SharedData: array<vec4<f32>, 256>;
var<workgroup> s_SharedVelocity: array<vec4<f32>, 256>;
var<workgroup> s_SharedSpin: array<vec4<f32>, 256>;
var<workgroup> s_SharedContacts: array<f32, 256>;
var<workgroup> s_SharedHeat: array<f32, 256>;
var<workgroup> s_SharedRadius: array<f32, 256>;

fn MergeSpeed(heat: f32, otherHeat: f32) -> f32 {
    if (sim.mergeSpeed <= 0.0) {
        return 0.0;
    }
    let glow = 1.0 - exp(-0.5 * (max(heat, 0.0) + max(otherHeat, 0.0)));
    return sim.mergeSpeed + sim.hotMergeSpeed * glow;
}

fn Softening(a: f32, b: f32) -> f32 {
    let mean = 0.5 * (a + b);
    return select(SOFTEN, max(SOFTEN, mean * mean), sim.sizeSoftening != 0);
}

// First pass: the particles' velocities, from the gravity of the others.
@compute @workgroup_size(256, 1, 1)
fn calculate_cs(@builtin(global_invocation_id) globalId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
    let index = globalId.x;
    let particleCount = u32(sim.particleCount);
    // Out-of-range threads still take part in the group's loads and barriers.
    let isActive = index < particleCount;

    var position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var trueRadius = 0.0;
    if (isActive) {
        position = u_Particles[index].pos;
        trueRadius = u_Particles[index].marker.w;
    }
    var acceleration = vec3<f32>(0.0, 0.0, 0.0);

    let stride = select(SHARED_DATA_SIZE, WORKGROUP_SIZE, sim.fullGravity != 0);
    for (var i = 0u; i < particleCount; i += stride) {
        if (i + localId.x < particleCount) {
            s_SharedData[localId.x] = u_Particles[i + localId.x].pos;
            s_SharedRadius[localId.x] = u_Particles[i + localId.x].marker.w;
        } else {
            s_SharedData[localId.x] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            s_SharedRadius[localId.x] = 0.0;
        }
        workgroupBarrier();

        for (var j = 0u; j < WORKGROUP_SIZE; j++) {
            let other = s_SharedData[j];
            let len = other.xyz - position.xyz;
            acceleration += sim.gravity * len * other.w
                / pow(dot(len, len) + Softening(trueRadius, s_SharedRadius[j]), POWER);
        }
        workgroupBarrier();
    }

    // A particle merged away (negative radius) stays where it was absorbed.
    if (!isActive || u_Particles[index].info.x < 0.0) {
        return;
    }

    // Kept for collide_cs (which then writes the whole Contact).
    u_Contacts[index].velocity = vec4<f32>(acceleration, 0.0);
    var vel = u_Particles[index].vel;
    vel = vec4<f32>(vel.xyz + sim.deltaT * sim.timeFactor * acceleration, vel.w);
    // Gradient texture position
    vel.w += 0.1 * sim.timeFactor * sim.deltaT;
    if (vel.w > 1.0) {
        vel.w -= 1.0;
    }
    u_Particles[index].vel = vel;
}

struct ContactTest {
    hit: bool,
    overlapping: bool,
    t: f32,
};

// Whether a pair touches this step: overlapping now, or apart but closing in so fast that it would
// cross within the step. Then t is the first touch, |offset + motion * t| = reach for t in [0, 1].
fn FindContact(offset: vec3<f32>, relativeVelocity: vec3<f32>, reach: f32, dt: f32) -> ContactTest {
    var result = ContactTest(false, false, 0.0);
    let distanceSquared = dot(offset, offset);
    result.overlapping = distanceSquared < reach * reach;
    if (distanceSquared < 1e-12) {
        return result;
    }
    if (result.overlapping) {
        result.hit = true;
        return result;
    }

    let motion = dt * relativeVelocity;
    let a = dot(motion, motion);
    let b = dot(offset, motion);
    let c = distanceSquared - reach * reach;
    let discriminant = b * b - a * c;
    // Closing in (b < 0) and the line comes within reach (discriminant >= 0).
    if (b >= 0.0 || discriminant < 0.0) {
        return result;
    }
    result.t = (-b - sqrt(discriminant)) / a;
    result.hit = result.t <= 1.0;
    return result;
}

// Loads WORKGROUP_SIZE particles' positions, velocities and radii into the shared arrays, from i on.
fn LoadShared(i: u32, localIndex: u32, particleCount: u32) {
    if (i + localIndex < particleCount) {
        let other = i + localIndex;
        s_SharedData[localIndex] = u_Particles[other].pos;
        s_SharedVelocity[localIndex] = vec4<f32>(u_Particles[other].vel.xyz, u_Particles[other].info.x);
        s_SharedSpin[localIndex] = u_Particles[other].spin;
    }
}

// Second pass: how many particles each one touches this step, into info.z for collide_cs. Each
// contact counts by how much it moves the particle: min(1, 2 share).
@compute @workgroup_size(256, 1, 1)
fn count_cs(@builtin(global_invocation_id) globalId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
    let index = globalId.x;
    let particleCount = u32(sim.particleCount);
    let isActive = index < particleCount;

    var position = vec3<f32>(0.0, 0.0, 0.0);
    var mass = 0.0;
    var velocity = vec3<f32>(0.0, 0.0, 0.0);
    var radius = 0.0;
    if (isActive) {
        position = u_Particles[index].pos.xyz;
        mass = u_Particles[index].pos.w;
        velocity = u_Particles[index].vel.xyz;
        radius = u_Particles[index].info.x;
    }
    let dt = sim.deltaT * sim.timeFactor;

    var contactCount = 0.0;
    for (var i = 0u; i < particleCount; i += WORKGROUP_SIZE) {
        LoadShared(i, localId.x, particleCount);
        workgroupBarrier();

        for (var j = 0u; j < WORKGROUP_SIZE && i + j < particleCount; j++) {
            if (i + j != index && radius >= 0.0 && s_SharedVelocity[j].w >= 0.0) {
                let contact = FindContact(position - s_SharedData[j].xyz, velocity - s_SharedVelocity[j].xyz,
                    radius + s_SharedVelocity[j].w, dt);
                if (contact.hit) {
                    let share = s_SharedData[j].w / (mass + s_SharedData[j].w);
                    contactCount += min(1.0, 2.0 * share);
                }
            }
        }
        workgroupBarrier();
    }

    if (isActive) {
        u_Particles[index].info.z = contactCount;
    }
}

// The heat a pair's loss gives this particle: the pair's relative speed along one direction goes
// from before to after, which loses 1/2 m_eff (before^2 - after^2).
fn PairHeat(share: f32, massFactor: f32, before: f32, after: f32) -> f32 {
    return sim.heatFactor * 0.5 * share * share * massFactor * max(before * before - after * after, 0.0);
}

// The pair's relative speed along the normal after this particle's part of a bounce.
fn AfterBounce(approach: f32, restitution: f32, share: f32, impulseShare: f32) -> f32 {
    return approach - (1.0 + restitution) * approach * (impulseShare / share);
}

// The bounciness at an approach speed (approach is negative): restitution up to restitutionSpeed,
// above it restitution (restitutionSpeed / speed)^(1/4) (Johnson's law for spheres).
fn RestitutionAt(approach: f32) -> f32 {
    let speed = -approach;
    if (sim.restitutionSpeed <= 0.0 || speed <= sim.restitutionSpeed) {
        return sim.restitution;
    }
    return sim.restitution * sqrt(sqrt(sim.restitutionSpeed / speed));
}

// Rolling resistance at a contact, for the particle that looks at the pair: a couple (equal and
// opposite torques, so the angular momentum is kept) against the two spins' difference.
fn ApplyRolling(radius: f32, otherRadius: f32, mass: f32, otherMass: f32, spin: vec4<f32>, otherSpin: vec4<f32>,
    normalImpulse: f32, share: f32, contactsMax: f32, spinChange: ptr<function, vec3<f32>>, heat: ptr<function, f32>) {
    if (sim.rollingResistance <= 0.0) {
        return;
    }
    let difference = spin.xyz - otherSpin.xyz;
    let rate = length(difference);
    if (rate < 1e-6) {
        return;
    }
    let inertia = mass * spin.w;
    let otherInertia = otherMass * otherSpin.w;
    let pairInertia = inertia * otherInertia / (inertia + otherInertia);
    let contactRadius = radius * otherRadius / (radius + otherRadius);
    let couple = min(sim.rollingResistance * contactRadius * normalImpulse, pairInertia * rate / contactsMax);
    *spinChange -= couple * difference / (rate * inertia);
    let after = rate - couple / pairInertia;
    *heat += sim.heatFactor * share * 0.5 * pairInertia * max(rate * rate - after * after, 0.0) / mass;
}

// Coulomb friction at a contact of two particles, for the particle that looks at the pair: normal
// points from the other to it, offset is from the other's center to its own.
fn ApplyFriction(normal: vec3<f32>, offset: vec3<f32>, reach: f32, relativeVelocity: vec3<f32>, approach: f32,
    restitution: f32, share: f32, impulseShare: f32, radius: f32, spin: vec4<f32>, otherRadius: f32,
    otherSpin: vec4<f32>, velocityChange: ptr<function, vec3<f32>>, spinChange: ptr<function, vec3<f32>>,
    heat: ptr<function, f32>) {
    // From each center to the contact point.
    let lever = -offset * (radius / reach);
    let otherLever = offset * (otherRadius / reach);
    let slip = relativeVelocity + cross(spin.xyz, lever) - cross(otherSpin.xyz, otherLever);
    let tangential = slip - dot(slip, normal) * normal;
    let slipSpeed = length(tangential);
    if (slipSpeed < 1e-6) {
        return;
    }

    // The other's share of the pair is 1 - share.
    let factor = 1.0 + share * dot(lever, lever) / spin.w
        + (1.0 - share) * dot(otherLever, otherLever) / otherSpin.w;
    let strength = min(sim.friction * (1.0 + restitution) * -approach, slipSpeed / factor);
    let direction = tangential / slipSpeed;
    *velocityChange -= impulseShare * strength * direction;
    // The torque lever x impulse over the inertia, per unit of the particle's mass.
    *spinChange -= impulseShare * strength * cross(lever, direction) / spin.w;
    *heat += PairHeat(share, 1.0 / factor, slipSpeed, slipSpeed - factor * strength * (impulseShare / share));
}

// Keeps the hardest impact over fragmentSpeed of a particle that can break: the halves part along
// the contact normal at a speed whose kinetic energy is half the heat the impact gives the particle.
fn RecordImpact(normal: vec3<f32>, approach: f32, restitution: f32, share: f32, impulseShare: f32, radius: f32,
    halvesMergeSpeed: f32, hardestImpact: ptr<function, f32>, fragmentVelocity: ptr<function, vec3<f32>>,
    fragmenting: ptr<function, bool>) {
    if (sim.fragmentSpeed <= 0.0 || radius < FRAGMENT_MIN_RADIUS || radius > FRAGMENT_MAX_RADIUS
        || -approach <= *hardestImpact) {
        return;
    }
    let after = AfterBounce(approach, restitution, share, impulseShare);
    let parting = share * sqrt(2.0 * max(approach * approach - after * after, 0.0));
    // Halves parting slower than their merge speed would merge again at once: no break.
    if (halvesMergeSpeed > 0.0 && parting <= halvesMergeSpeed) {
        return;
    }
    *hardestImpact = -approach;
    *fragmentVelocity = normal * parting;
    *fragmenting = true;
}

// The energy the contact springs hold between two particles overlapping by overlap (of their reach).
fn SpringEnergy(overlap: f32, reach: f32, mu: f32, dt: f32) -> f32 {
    if (overlap <= 0.0 || dt <= 0.0) {
        return 0.0;
    }
    let stiffness = mu * CONTACT_STIFFNESS / (dt * dt);
    let cap = CONTACT_OVERLAP_MAX * reach;
    let inner = min(overlap, cap);
    return 0.5 * stiffness * inner * inner + stiffness * cap * max(overlap - cap, 0.0);
}

// The spring energy between a particle (position, radius, mass) and another.
fn SpringEnergyWith(position: vec3<f32>, radius: f32, mass: f32, other: Particle, dt: f32) -> f32 {
    let reach = radius + other.info.x;
    let overlap = reach - length(position - other.pos.xyz);
    return SpringEnergy(overlap, reach, mass * other.pos.w / (mass + other.pos.w), dt);
}

// Whether particle a (mass and index) takes the other in a merge: more mass, or the lower index.
fn Dominates(massA: f32, indexA: u32, massB: f32, indexB: u32) -> bool {
    return massA > massB || (massA == massB && indexA < indexB);
}

// Third pass: collisions, against every other particle. Each particle reads the others' positions
// and velocities as calculate_cs left them and writes only its own Contact, so the order the
// groups run in doesn't matter; integrate_cs applies them.
@compute @workgroup_size(256, 1, 1)
fn collide_cs(@builtin(global_invocation_id) globalId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
    let index = globalId.x;
    let particleCount = u32(sim.particleCount);
    let isActive = index < particleCount;

    var position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var velocity = vec3<f32>(0.0, 0.0, 0.0);
    var spin = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var radius = 0.0;
    var contactCount = 0.0;
    var ownHeat = 0.0;
    var trueRadius = 0.0;
    var gravityPull = vec3<f32>(0.0, 0.0, 0.0);
    if (isActive) {
        position = u_Particles[index].pos;
        velocity = u_Particles[index].vel.xyz;
        spin = u_Particles[index].spin;
        radius = u_Particles[index].info.x;
        contactCount = u_Particles[index].info.z;
        // Its heat, for the merge speeds; its halves, if it breaks, merge as it would with itself.
        ownHeat = u_Particles[index].info.y;
        trueRadius = u_Particles[index].marker.w;
        // The gravity's pull on it this step (calculate_cs).
        gravityPull = u_Contacts[index].velocity.xyz;
    }
    let ownMergeSpeed = MergeSpeed(ownHeat, ownHeat);
    // The room it has to grow into: the nearest other particle's surface.
    var clearance = 1.0e30;
    // A merged particle short of its true size can also grow into its neighbors, by GROWTH_PER_STEP.
    let pushedRadius = min(trueRadius, radius + GROWTH_PER_STEP * trueRadius);
    var growthEnergy = 0.0;

    // integrate_cs's step: how far a velocity moves a particle this frame.
    let dt = sim.deltaT * sim.timeFactor;

    var velocityChange = vec3<f32>(0.0, 0.0, 0.0);
    var spinChange = vec3<f32>(0.0, 0.0, 0.0);
    // Position corrections of the swept contacts.
    var sweepShift = vec3<f32>(0.0, 0.0, 0.0);
    // With the contact springs off: the overlaps' push-out.
    var shift = vec3<f32>(0.0, 0.0, 0.0);
    var heat = 0.0;
    var springPushes = vec3<f32>(0.0, 0.0, 0.0);
    var impulseSquares = 0.0;
    var spinSquares = 0.0;
    // Merging: the most massive of the touching particles that dominate this one (or itself).
    var merging = false;
    var mergeTarget = index;
    var mergeTargetMass = position.w;
    // Fragmenting: the hardest impact over fragmentSpeed, its normal and the speed the halves part at.
    var hardestImpact = sim.fragmentSpeed;
    var fragmentVelocity = vec3<f32>(0.0, 0.0, 0.0);
    var fragmenting = false;

    for (var i = 0u; i < particleCount; i += WORKGROUP_SIZE) {
        LoadShared(i, localId.x, particleCount);
        if (i + localId.x < particleCount) {
            s_SharedContacts[localId.x] = u_Particles[i + localId.x].info.z;
            s_SharedHeat[localId.x] = u_Particles[i + localId.x].info.y;
        }
        workgroupBarrier();

        for (var j = 0u; j < WORKGROUP_SIZE && i + j < particleCount; j++) {
            if (i + j == index || radius < 0.0 || s_SharedVelocity[j].w < 0.0) {
                continue;
            }
            // Closing in, a neighbor counts where it gets to this step.
            let apart = position.xyz - s_SharedData[j].xyz;
            let gap = length(apart);
            var closing = 0.0;
            if (gap > 0.0) {
                closing = min(dot(velocity - s_SharedVelocity[j].xyz, apart) / gap, 0.0);
            }
            clearance = min(clearance, gap + dt * closing - s_SharedVelocity[j].w);
            if (pushedRadius > radius && sim.contactSprings != 0 && dt > 0.0) {
                // Not with a slow pair: it merges when they touch, and has no spring.
                let growRadius = s_SharedVelocity[j].w;
                let merges = length(velocity - s_SharedVelocity[j].xyz) < MergeSpeed(ownHeat, s_SharedHeat[j]);
                if (!merges) {
                    let mu = position.w * s_SharedData[j].w / (position.w + s_SharedData[j].w);
                    growthEnergy += SpringEnergy(pushedRadius + growRadius - gap, pushedRadius + growRadius, mu, dt)
                        - SpringEnergy(radius + growRadius - gap, radius + growRadius, mu, dt);
                }
            }

            let otherPosition = s_SharedData[j];
            let otherVelocity = s_SharedVelocity[j];

            // From the other particle to this one.
            let offset = position.xyz - otherPosition.xyz;
            let reach = radius + otherVelocity.w;
            let relativeVelocity = velocity - otherVelocity.xyz;
            let contact = FindContact(offset, relativeVelocity, reach, dt);
            if (!contact.hit) {
                continue;
            }

            // This particle's share of the pair's change: the other's mass over the total.
            let share = otherPosition.w / (position.w + otherPosition.w);
            // And the pair's part of the busier particle's contacts (both counted this one).
            let contactsMax = max(max(contactCount, s_SharedContacts[j]), 1.0);
            let impulseShare = share / contactsMax;
            let velocityBefore = velocityChange;
            let spinBefore = spinChange;

            if (!contact.overlapping) {
                // Apart now: bounce the pair at the first touch, with the contact normal, approach
                // speed and offset there.
                let touchOffset = offset + dt * relativeVelocity * contact.t;
                let normal = touchOffset / reach;
                let approach = dot(relativeVelocity, normal);
                if (approach >= 0.0) {
                    continue;
                }
                let restitution = RestitutionAt(approach);
                RecordImpact(normal, approach, restitution, share, impulseShare, trueRadius, ownMergeSpeed,
                    &hardestImpact, &fragmentVelocity, &fragmenting);
                let bounce = -(1.0 + restitution) * impulseShare * approach;
                velocityChange += bounce * normal;
                heat += PairHeat(share, 1.0, approach, AfterBounce(approach, restitution, share, impulseShare));
                ApplyFriction(normal, touchOffset, reach, relativeVelocity, approach, restitution, share, impulseShare,
                    radius, spin, otherVelocity.w, s_SharedSpin[j], &velocityChange, &spinChange, &heat);
                ApplyRolling(radius, otherVelocity.w, position.w, otherPosition.w, spin, s_SharedSpin[j],
                    position.w * abs(dot(velocityChange - velocityBefore, normal)), share, contactsMax,
                    &spinChange, &heat);
                // integrate_cs moves the particle the whole step with the new velocity, but it had
                // the old one until the touch: take back that part of the step.
                sweepShift -= contact.t * dt * (velocityChange - velocityBefore);
                impulseSquares += dot(velocityChange - velocityBefore, velocityChange - velocityBefore);
                spinSquares += dot(spinChange - spinBefore, spinChange - spinBefore);
                continue;
            }

            // Touching slowly enough, the two merge instead (absorb_cs, kill_cs): no bounce or push.
            if (length(relativeVelocity) < MergeSpeed(ownHeat, s_SharedHeat[j])) {
                merging = true;
                if (Dominates(otherPosition.w, i + j, position.w, index)
                    && (mergeTarget == index || Dominates(otherPosition.w, i + j, mergeTargetMass, mergeTarget))) {
                    mergeTarget = i + j;
                    mergeTargetMass = otherPosition.w;
                }
                continue;
            }

            let dist = length(offset);
            let normal = offset / dist;

            // Negative while the two close in. Pairs already separating keep their velocities.
            let approach = dot(relativeVelocity, normal);
            if (approach < 0.0) {
                // With the push-outs (contact springs off), an overlapping pair still closing in
                // stops along the normal instead of bouncing.
                let restitution = select(0.0, RestitutionAt(approach), sim.contactSprings != 0);
                RecordImpact(normal, approach, restitution, share, impulseShare, trueRadius, ownMergeSpeed,
                    &hardestImpact, &fragmentVelocity, &fragmenting);
                let bounce = -(1.0 + restitution) * impulseShare * approach;
                velocityChange += bounce * normal;
                heat += PairHeat(share, 1.0, approach, AfterBounce(approach, restitution, share, impulseShare));
                ApplyFriction(normal, offset, reach, relativeVelocity, approach, restitution, share, impulseShare,
                    radius, spin, otherVelocity.w, s_SharedSpin[j], &velocityChange, &spinChange, &heat);
            }

            if (sim.contactSprings == 0) {
                shift += SEPARATION * share * (reach - dist) * normal;
            } else if (dt > 0.0) {
                let overlap = min(reach - dist, CONTACT_OVERLAP_MAX * reach);
                // Damped while they part (approach > 0), divided among the contacts (impulseShare).
                var damping = 0.0;
                if (approach > 0.0) {
                    damping = 2.0 * CONTACT_DAMPING * sqrt(CONTACT_STIFFNESS) * approach;
                }
                let push = share * CONTACT_STIFFNESS * overlap / dt * normal;
                springPushes += push;
                velocityChange += push - impulseShare * damping * normal;
                // The parting speed it takes is heat; only while they part.
                if (damping > 0.0) {
                    heat += PairHeat(share, 1.0, approach, approach - (impulseShare / share) * damping);
                }
                impulseSquares -= 2.0 * dot(velocityChange - velocityBefore, push) - dot(push, push);
            }
            ApplyRolling(radius, otherVelocity.w, position.w, otherPosition.w, spin, s_SharedSpin[j],
                position.w * abs(dot(velocityChange - velocityBefore, normal)), share, contactsMax,
                &spinChange, &heat);
            impulseSquares += dot(velocityChange - velocityBefore, velocityChange - velocityBefore);
            spinSquares += dot(spinChange - spinBefore, spinChange - spinBefore);
        }
        workgroupBarrier();
    }

    if (!isActive) {
        return;
    }
    // The impulses' cross terms: their energy, per unit of mass, comes out of the heat.
    let impulses = velocityChange - springPushes;
    heat -= sim.heatFactor * 0.5 * (dot(impulses, impulses) - impulseSquares
        + spin.w * (dot(spinChange, spinChange) - spinSquares));

    // The radius integrate_cs gives it (info.w). A merged particle short of its true size grows into
    // the free room around it for nothing; in a pile, by GROWTH_PER_STEP into its neighbors, paying
    // the energy that gives their springs from its heat. Not while it merges.
    if (radius >= 0.0) {
        var grown = select(max(radius, min(trueRadius, clearance)), radius, merging);
        if (!merging && pushedRadius > grown) {
            var heatEnergy = 0.0;
            if (sim.heatFactor > 0.0) {
                heatEnergy = (u_Particles[index].info.y + heat) * position.w / sim.heatFactor;
            }
            if (sim.heating == 0) {
                grown = pushedRadius;
            } else if (growthEnergy <= heatEnergy) {
                grown = pushedRadius;
                heat -= sim.heatFactor * growthEnergy / position.w;
            }
        }
        u_Particles[index].info.w = grown;
    }

    // In a dense pile the push-outs of many neighbors add up: at most a radius per step.
    let shiftLength = length(shift);
    if (shiftLength > radius) {
        shift *= radius / shiftLength;
    }
    // A push-out moves the particle without a velocity change: its heat pays for the work it does
    // against the gravity.
    heat += sim.heatFactor * dot(gravityPull, shift);
    var contactOut: Contact;
    contactOut.velocity = vec4<f32>(velocityChange, select(0.0, heat, sim.heating != 0));
    contactOut.shift = vec4<f32>(shift + sweepShift, select(0.0, 1.0, merging));
    contactOut.spin = vec4<f32>(spinChange, f32(mergeTarget));
    // A particle merging this step doesn't also break.
    contactOut.fragment = vec4<f32>(fragmentVelocity, select(0.0, 1.0, fragmenting && !merging));
    u_Contacts[index] = contactOut;
}

// The gravity's potential energy between two particles a distance apart, from their merged state.
// A merge releases it, a break creates it.
fn PairPotential(dist: f32, mass: f32, otherMass: f32, soften: f32) -> f32 {
    return 2.0 * sim.gravity * mass * otherMass
        * (sqrt(sqrt(dist * dist + soften)) - sqrt(sqrt(soften)));
}

// A particle's merge: it absorbs the particles that name it as their target, when it is its own
// (nothing touching it dominates it). The survivor takes the sum of the masses and of the true
// volumes, the mass-weighted position, velocity and heat, and a spin that keeps the angular
// momentum about the new center. What the merge loses (motion, spins, gravity's potential energy,
// the springs the merging particles had) is heat.
@compute @workgroup_size(256, 1, 1)
fn absorb_cs(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let index = globalId.x;
    let particleCount = u32(sim.particleCount);
    if (index >= particleCount) {
        return;
    }

    let own = u_Contacts[index];
    var particle = u_Particles[index];
    if (own.shift.w == 0.0 || u32(own.spin.w) != index || particle.info.x < 0.0) {
        return;
    }

    var mass = particle.pos.w;
    var position = particle.pos.xyz + own.shift.xyz;
    var velocity = particle.vel.xyz + own.velocity.xyz;
    var spin = particle.spin.xyz + own.spin.xyz;
    var inertia = particle.spin.w;
    var heat = particle.info.y + own.velocity.w;
    var radius = particle.info.x;
    var trueVolume = particle.marker.w * particle.marker.w * particle.marker.w;
    var absorbed = false;
    // The survivor as it was, for the contact springs it had with the others.
    let survivorPosition = position;
    let survivorRadius = radius;
    let survivorMass = mass;
    let dt = sim.deltaT * sim.timeFactor;
    let springs = sim.contactSprings != 0 && dt > 0.0;
    // The energy in the springs the merging particles had with the others, which the merge ends.
    var springsBefore = 0.0;

    for (var j = 0u; j < particleCount; j++) {
        let other = u_Contacts[j];
        if (j == index || other.shift.w == 0.0 || u32(other.spin.w) != index) {
            continue;
        }
        let victim = u_Particles[j];
        if (victim.info.x < 0.0) {
            continue;
        }

        let victimMass = victim.pos.w;
        let victimPosition = victim.pos.xyz + other.shift.xyz;
        let victimVelocity = victim.vel.xyz + other.velocity.xyz;
        let victimSpin = victim.spin.xyz + other.spin.xyz;
        let victimInertia = victim.spin.w;
        let victimHeat = victim.info.y + other.velocity.w;
        let victimRadius = victim.info.x;

        let total = mass + victimMass;
        heat = (mass * heat + victimMass * victimHeat) / total;
        let newPosition = (mass * position + victimMass * victimPosition) / total;
        if (sim.heating != 0) {
            // The gravity's potential energy the merge releases, only with gravity between all pairs:
            // the sampled one has no potential energy.
            let potential = sim.fullGravity != 0;
            // The true radii: the absorbed so far's, the victim's, and the two's merged.
            let partRadius = pow(trueVolume, 1.0 / 3.0);
            let victimTrue = victim.marker.w;
            let mergedRadius = pow(trueVolume + victimTrue * victimTrue * victimTrue, 1.0 / 3.0);
            var released = 0.0;
            if (potential) {
                released = PairPotential(length(victimPosition - position), mass, victimMass, Softening(partRadius, victimTrue));
            }
            for (var k = 0u; k < particleCount; k++) {
                if (k == index || k == j) {
                    continue;
                }
                let neighbor = u_Particles[k].pos;
                if (u_Particles[k].info.x < 0.0) {
                    continue;
                }
                let neighborContact = u_Contacts[k];
                let neighborMerging = neighborContact.shift.w != 0.0 && u32(neighborContact.spin.w) == index;
                if (k < j && neighborMerging) {
                    continue;
                }
                if (potential) {
                    let neighborRadius = u_Particles[k].marker.w;
                    released += PairPotential(length(position - neighbor.xyz), mass, neighbor.w, Softening(partRadius, neighborRadius))
                        + PairPotential(length(victimPosition - neighbor.xyz), victimMass, neighbor.w, Softening(victimTrue, neighborRadius))
                        - PairPotential(length(newPosition - neighbor.xyz), total, neighbor.w, Softening(mergedRadius, neighborRadius));
                }
                if (springs && !neighborMerging) {
                    springsBefore += SpringEnergyWith(victimPosition, victimRadius, victimMass, u_Particles[k], dt);
                }
            }
            heat += sim.heatFactor * released / total;
        }
        let newVelocity = (mass * velocity + victimMass * victimVelocity) / total;
        let newRadius = max(radius, victimRadius);

        // The angular momentum about the new center: the two bodies' orbits about it, and their own
        // spins, all go into the merged particle's spin.
        let angularMomentum =
            mass * cross(position - newPosition, velocity - newVelocity)
            + victimMass * cross(victimPosition - newPosition, victimVelocity - newVelocity)
            + mass * inertia * spin
            + victimMass * victimInertia * victimSpin;
        let newInertia = (mass * (inertia + dot(position - newPosition, position - newPosition))
            + victimMass * (victimInertia + dot(victimPosition - newPosition, victimPosition - newPosition))) / total;
        let newSpin = angularMomentum / (total * newInertia);
        if (sim.heating != 0) {
            let before = mass * dot(velocity, velocity) + victimMass * dot(victimVelocity, victimVelocity)
                + mass * inertia * dot(spin, spin) + victimMass * victimInertia * dot(victimSpin, victimSpin);
            let after = total * dot(newVelocity, newVelocity) + total * newInertia * dot(newSpin, newSpin);
            // Never below 0 but for rounding.
            heat += sim.heatFactor * 0.5 * max(before - after, 0.0) / total;
        }

        spin = newSpin;
        inertia = newInertia;
        position = newPosition;
        velocity = newVelocity;
        radius = newRadius;
        trueVolume += victim.marker.w * victim.marker.w * victim.marker.w;
        mass = total;
        absorbed = true;
    }

    if (!absorbed) {
        // Touching slowly, but nothing to absorb this step (its partners wait, or merge into
        // another): it grows where it is, as collide_cs grows a particle that doesn't merge.
        let ownTrue = particle.marker.w;
        let pushedAlone = min(ownTrue, radius + GROWTH_PER_STEP * ownTrue);
        if (pushedAlone <= radius) {
            return;
        }
        var clearanceAlone = ownTrue;
        var growthEnergyAlone = 0.0;
        for (var k = 0u; k < particleCount; k++) {
            let neighbor = u_Particles[k];
            if (k == index || neighbor.info.x < 0.0) {
                continue;
            }
            clearanceAlone = min(clearanceAlone, length(particle.pos.xyz - neighbor.pos.xyz) - neighbor.info.x);
            // Not with a slow pair, as in collide_cs.
            if (springs && length(particle.vel.xyz - neighbor.vel.xyz) >= MergeSpeed(particle.info.y, neighbor.info.y)) {
                growthEnergyAlone += SpringEnergyWith(particle.pos.xyz, pushedAlone, mass, neighbor, dt)
                    - SpringEnergyWith(particle.pos.xyz, radius, mass, neighbor, dt);
            }
        }
        var grownAlone = max(radius, min(ownTrue, clearanceAlone));
        if (pushedAlone > grownAlone) {
            var heatEnergyAlone = 0.0;
            if (sim.heatFactor > 0.0) {
                heatEnergyAlone = max(heat, 0.0) * mass / sim.heatFactor;
            }
            if (sim.heating == 0) {
                grownAlone = pushedAlone;
            } else if (growthEnergyAlone <= heatEnergyAlone) {
                grownAlone = pushedAlone;
                u_Contacts[index].velocity.w -= sim.heatFactor * growthEnergyAlone / mass;
            }
        }
        u_Particles[index].info.w = grownAlone;
        return;
    }

    // At the center of mass the merged sphere can reach into particles none of the merging ones
    // touched, so it takes no more than the room there, up to the nearest other particle's surface
    // (and its true radius); then this step's growth into its neighbors.
    let trueRadius = pow(trueVolume, 1.0 / 3.0);
    var clearance = trueRadius;
    for (var k = 0u; k < particleCount; k++) {
        let neighbor = u_Particles[k];
        if (k == index || neighbor.info.x < 0.0) {
            continue;
        }
        let neighborContact = u_Contacts[k];
        if (neighborContact.shift.w != 0.0 && u32(neighborContact.spin.w) == index) {
            continue;
        }
        clearance = min(clearance, length(position - neighbor.pos.xyz) - neighbor.info.x);
        if (springs && sim.heating != 0) {
            springsBefore += SpringEnergyWith(survivorPosition, survivorRadius, survivorMass, neighbor, dt);
        }
    }
    // Never vanishing: below a quarter of its size it still overlaps, and pays for that below.
    let fitted = max(clearance, 0.25 * radius);
    let pushed = min(trueRadius, fitted + GROWTH_PER_STEP * trueRadius);
    var springsAfter = 0.0;
    var growthEnergy = 0.0;
    if (springs && ((sim.heating != 0 && fitted > clearance) || pushed > fitted)) {
        for (var k = 0u; k < particleCount; k++) {
            let neighbor = u_Particles[k];
            if (k == index || neighbor.info.x < 0.0) {
                continue;
            }
            let neighborContact = u_Contacts[k];
            if (neighborContact.shift.w != 0.0 && u32(neighborContact.spin.w) == index) {
                continue;
            }
            let fittedEnergy = SpringEnergyWith(position, fitted, mass, neighbor, dt);
            springsAfter += fittedEnergy;
            // Not with a slow pair, as in collide_cs.
            if (length(velocity - neighbor.vel.xyz) >= MergeSpeed(heat, neighbor.info.y)) {
                growthEnergy += SpringEnergyWith(position, pushed, mass, neighbor, dt) - fittedEnergy;
            }
        }
    }
    radius = fitted;
    if (sim.heating != 0) {
        heat += sim.heatFactor * (springsBefore - springsAfter) / mass;
        var heatEnergy = 0.0;
        if (sim.heatFactor > 0.0) {
            heatEnergy = max(heat, 0.0) * mass / sim.heatFactor;
        }
        if (pushed > fitted && growthEnergy <= heatEnergy) {
            heat -= sim.heatFactor * growthEnergy / mass;
            radius = pushed;
        }
    } else {
        radius = pushed;
    }

    particle.pos = vec4<f32>(position, mass);
    particle.vel = vec4<f32>(velocity, particle.vel.w);
    particle.spin = vec4<f32>(spin, inertia);
    particle.info.x = radius;
    particle.info.y = heat;
    // collide_cs's growth was for where it was: none more this step.
    particle.info.w = radius;
    particle.marker.w = trueRadius;
    u_Particles[index] = particle;

    // Spent (integrate_cs adds them): keep the flag and the target, which kill_cs reads.
    var spent = own;
    spent.velocity = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    spent.shift = vec4<f32>(0.0, 0.0, 0.0, spent.shift.w);
    spent.spin = vec4<f32>(0.0, 0.0, 0.0, spent.spin.w);
    u_Contacts[index] = spent;
}

// After absorb_cs: the particles that were absorbed are gone (a negative radius, no mass, nothing
// drawn or collided with).
@compute @workgroup_size(256, 1, 1)
fn kill_cs(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let index = globalId.x;
    if (index >= u32(sim.particleCount)) {
        return;
    }

    let own = u_Contacts[index];
    let victimOf = u32(own.spin.w);
    if (own.shift.w == 0.0 || victimOf == index) {
        return;
    }
    // Its target is merging into another: this one waits for the next step.
    if (u32(u_Contacts[victimOf].spin.w) != victimOf) {
        return;
    }

    var particle = u_Particles[index];
    particle.pos.w = 0.0;
    particle.vel = vec4<f32>(0.0, 0.0, 0.0, particle.vel.w);
    particle.spin = vec4<f32>(0.0, 0.0, 0.0, particle.spin.w);
    particle.info = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    u_Particles[index] = particle;
}

// Last pass: the collisions, then Euler integration of the positions.
@compute @workgroup_size(256, 1, 1)
fn integrate_cs(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let index = globalId.x;
    if (index >= u32(sim.particleCount)) {
        return;
    }

    var particle = u_Particles[index];
    if (particle.info.x < 0.0) {
        return;
    }

    if (sim.collisions != 0) {
        let contact = u_Contacts[index];
        particle.vel = vec4<f32>(particle.vel.xyz + contact.velocity.xyz, particle.vel.w);
        particle.pos = vec4<f32>(particle.pos.xyz + contact.shift.xyz, particle.pos.w);
        particle.info.y = max(particle.info.y + contact.velocity.w, 0.0);
        particle.spin = vec4<f32>(particle.spin.xyz + contact.spin.xyz, particle.spin.w);

        // A merged particle grows towards its true radius (marker.w), as collide_cs chose (info.w).
        if (particle.marker.w > particle.info.x) {
            particle.info.x = min(particle.marker.w, max(particle.info.x, particle.info.w));
        }
    }

    particle.pos = vec4<f32>(particle.pos.xyz + sim.deltaT * sim.timeFactor * particle.vel.xyz, particle.pos.w);

    // The marker turns about the spin axis (Rodrigues' formula, exact for any step).
    let spinSpeed = length(particle.spin.xyz);
    if (spinSpeed > 1e-6) {
        let angle = spinSpeed * sim.deltaT * sim.timeFactor;
        let axis = particle.spin.xyz / spinSpeed;
        let marker = particle.marker.xyz;
        particle.marker = vec4<f32>(normalize(marker * cos(angle) + cross(axis, marker) * sin(angle)
            + axis * dot(axis, marker) * (1.0 - cos(angle))), particle.marker.w);
    }
    particle.info.y *= exp(-sim.deltaT / sim.coolSeconds);

    u_Particles[index] = particle;
}

// Before fragment_cs: the slots of the particles merged away (or left free), listed in u_Free.
@compute @workgroup_size(256, 1, 1)
fn collect_free_cs(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let index = globalId.x;
    if (index >= u32(sim.particleCount) || u_Particles[index].info.x >= 0.0) {
        return;
    }
    let slot = atomicAdd(&u_Free.count, 1u);
    u_Free.slots[slot] = index;
}

// After integrate_cs: a particle hit hard enough (collide_cs) breaks into two halves of half its
// mass and volume, touching at its center along the impact normal, parting at the speed collide_cs
// chose: one stays in its slot, the other takes a free one (none left: it doesn't break). The
// parting's kinetic energy, the spin's change and the gravity's potential energy between the halves
// come out of the particle's heat, as does the contact springs' energy with the neighbors the
// halves reach into: with too little heat (or heating off), it doesn't break.
@compute @workgroup_size(256, 1, 1)
fn fragment_cs(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let index = globalId.x;
    if (index >= u32(sim.particleCount)) {
        return;
    }
    let contact = u_Contacts[index];
    var particle = u_Particles[index];
    if (contact.fragment.w == 0.0 || particle.info.x < 0.0) {
        return;
    }

    // 2^(-1/3): half the volume.
    let radius = particle.info.x * 0.7937005;
    let speed = length(contact.fragment.xyz);
    var normal = vec3<f32>(1.0, 0.0, 0.0);
    if (speed > 1e-6) {
        normal = contact.fragment.xyz / speed;
    }
    let center = particle.pos.xyz;
    let velocity = particle.vel.xyz;
    // The halves are solid spheres (inertia 2/5 r^2 over the mass), with the particle's angular
    // momentum in their spins.
    let halfInertia = 0.4 * radius * radius;
    let halfSpin = particle.spin.xyz * particle.spin.w / halfInertia;
    let spinGain = 0.5 * particle.pos.w * (halfInertia * dot(halfSpin, halfSpin)
        - particle.spin.w * dot(particle.spin.xyz, particle.spin.xyz));
    let potential = sim.fullGravity != 0;
    let halfTrue = particle.marker.w * 0.7937005;
    var gained = 0.125 * particle.pos.w * speed * speed + spinGain;
    if (potential) {
        gained += PairPotential(2.0 * radius, 0.5 * particle.pos.w, 0.5 * particle.pos.w, Softening(halfTrue, halfTrue));
    }
    let dt = sim.deltaT * sim.timeFactor;
    let springs = sim.contactSprings != 0 && dt > 0.0;
    if (springs || potential) {
        let particleCount = u32(sim.particleCount);
        for (var j = 0u; j < particleCount; j++) {
            let other = u_Particles[j];
            if (j == index || other.info.x < 0.0) {
                continue;
            }
            if (springs) {
                gained += SpringEnergyWith(center + normal * radius, radius, 0.5 * particle.pos.w, other, dt)
                    + SpringEnergyWith(center - normal * radius, radius, 0.5 * particle.pos.w, other, dt)
                    - SpringEnergyWith(center, particle.info.x, particle.pos.w, other, dt);
            }
            if (potential) {
                let halfSoften = Softening(halfTrue, other.marker.w);
                gained += PairPotential(length(center + normal * radius - other.pos.xyz), 0.5 * particle.pos.w, other.pos.w, halfSoften)
                    + PairPotential(length(center - normal * radius - other.pos.xyz), 0.5 * particle.pos.w, other.pos.w, halfSoften)
                    - PairPotential(length(center - other.pos.xyz), particle.pos.w, other.pos.w, Softening(particle.marker.w, other.marker.w));
            }
        }
    }
    // Paid from the heat; with too little (or heating off), the particle doesn't break.
    var heatEnergy = 0.0;
    if (sim.heatFactor > 0.0) {
        heatEnergy = particle.info.y * particle.pos.w / sim.heatFactor;
    }
    if (sim.heating == 0 || gained > heatEnergy) {
        return;
    }
    particle.info.y -= sim.heatFactor * gained / particle.pos.w;

    let taken = atomicAdd(&u_Free.taken, 1u);
    if (taken >= atomicLoad(&u_Free.count)) {
        return;
    }
    let slot = u_Free.slots[taken];

    particle.pos = vec4<f32>(center + normal * radius, 0.5 * particle.pos.w);
    particle.vel = vec4<f32>(velocity + 0.5 * contact.fragment.xyz, particle.vel.w);
    particle.spin = vec4<f32>(halfSpin, halfInertia);
    particle.info.x = radius;
    particle.info.z = 0.0;
    particle.marker.w *= 0.7937005;
    u_Particles[index] = particle;

    particle.pos = vec4<f32>(center - normal * radius, particle.pos.w);
    particle.vel = vec4<f32>(velocity - 0.5 * contact.fragment.xyz, particle.vel.w);
    u_Particles[slot] = particle;
}
`;
