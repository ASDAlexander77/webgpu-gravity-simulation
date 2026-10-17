// The controls of compute_nbody_collisions (the example's ImGui window), as an HTML panel in the
// top right corner. Every feature has a switch; a slider beneath one sets how much, while it is on.
// Values are written back to the simulation as they change.

class ControlPanel {
    constructor(root, nbody) {
        this.root = root;
        this.nbody = nbody;
        // Every control's refresh, to show the simulation's values again (Defaults, the keys).
        this.refreshers = [];
    }

    refresh() {
        for (const update of this.refreshers) {
            update();
        }
    }

    element(tag, className, parent, text) {
        const e = document.createElement(tag);
        if (className) {
            e.className = className;
        }
        if (text !== undefined) {
            e.textContent = text;
        }
        if (parent) {
            parent.appendChild(e);
        }
        return e;
    }

    section(title, open) {
        const details = this.element('details', 'section', this.root);
        details.open = open;
        this.element('summary', '', details, title);
        return this.element('div', 'body', details);
    }

    // A switch. `sub` holds the controls that apply only while it is on (shown and hidden with it).
    checkbox(parent, label, get, set, hint) {
        const row = this.element('label', 'row check', parent);
        const input = this.element('input', '', row);
        input.type = 'checkbox';
        this.element('span', '', row, label);
        if (hint) {
            row.title = hint;
        }
        const sub = this.element('div', 'sub', parent);
        const update = () => {
            input.checked = get();
            sub.hidden = !input.checked;
        };
        input.addEventListener('change', () => {
            set(input.checked);
            this.nbody.applySwitches();
            update();
            input.blur();
        });
        this.refreshers.push(update);
        update();
        return sub;
    }

    slider(parent, label, get, set, min, max, step, decimals = 3) {
        const row = this.element('label', 'row slider', parent);
        this.element('span', 'name', row, label);
        const input = this.element('input', '', row);
        input.type = 'range';
        input.min = min;
        input.max = max;
        input.step = step;
        const value = this.element('span', 'value', row);
        const update = () => {
            input.value = get();
            value.textContent = Number(get()).toFixed(decimals);
        };
        input.addEventListener('input', () => {
            set(parseFloat(input.value));
            this.nbody.applySwitches();
            update();
        });
        this.refreshers.push(update);
        update();
        return row;
    }

    select(parent, label, options, get, set) {
        const row = this.element('label', 'row select', parent);
        this.element('span', 'name', row, label);
        const input = this.element('select', '', row);
        options.forEach((option, index) => {
            const e = this.element('option', '', input, option.label ?? option);
            e.value = index;
        });
        const update = () => {
            input.value = get();
        };
        input.addEventListener('change', () => {
            set(parseInt(input.value));
            update();
            input.blur();
        });
        this.refreshers.push(update);
        update();
        return row;
    }

    button(parent, label, onClick) {
        const button = this.element('button', '', parent, label);
        button.addEventListener('click', () => {
            onClick();
            button.blur();
        });
        return button;
    }

    // Like the sliders, but for a value of the simulation's constants.
    constant(parent, label, index, min, max, step, decimals) {
        const nbody = this.nbody;
        return this.slider(parent, label, () => nbody.f32[index], (v) => { nbody.f32[index] = v; }, min, max, step, decimals);
    }

    build() {
        const nb = this.nbody;
        const ui = nb.ui;

        const title = this.element('div', 'title', this.root);
        this.element('span', '', title, 'N-Body Collisions');
        const status = this.element('div', 'status', this.root);
        nb.onStatus = (text) => { status.textContent = text; };
        status.textContent = nb.statusText();

        // --- Simulation: each feature has a switch, a slider beneath sets how much -------------
        const sim = this.section('Simulation', true);
        this.checkbox(sim, 'Paused (Space)', () => nb.paused, (v) => { nb.paused = v; });
        this.checkbox(sim, 'Collisions (C)', () => nb.collisions, (v) => nb.setCollisions(v));
        this.checkbox(sim, 'Heating (H)', () => nb.heating, (v) => nb.setHeating(v));
        this.checkbox(sim, 'Gravity between all pairs', () => nb.fullGravity, (v) => nb.setFullGravity(v),
            'Every pair pulls both ways, so the momentum is kept; off, a body feels a quarter of the light particles');
        this.checkbox(sim, 'Contact springs (off: push-outs)', () => nb.contactSprings, (v) => { nb.contactSprings = v; });
        let sub = this.checkbox(sim, 'Friction', () => nb.frictionOn, (v) => { nb.frictionOn = v; });
        this.slider(sub, 'Friction mu', () => nb.frictionValue, (v) => { nb.frictionValue = v; }, 0, 1, 0.01, 2);
        sub = this.checkbox(sim, 'Rolling resistance', () => nb.rollingOn, (v) => { nb.rollingOn = v; });
        this.slider(sub, 'Rolling mu_r', () => nb.rollingValue, (v) => { nb.rollingValue = v; }, 0, ROLLING_RESISTANCE_MAX, 0.005, 3);
        sub = this.checkbox(sim, 'Hard hits bounce less', () => nb.restitutionSpeedOn, (v) => { nb.restitutionSpeedOn = v; });
        this.slider(sub, 'Above (u/s)', () => nb.restitutionSpeedValue, (v) => { nb.restitutionSpeedValue = v; }, 0, RESTITUTION_SPEED_MAX, 0.1, 1);
        this.checkbox(sim, 'Gravity softened by size', () => nb.sizeSoftening, (v) => { nb.sizeSoftening = v; });
        sub = this.checkbox(sim, 'Merging', () => nb.mergeOn, (v) => { nb.mergeOn = v; });
        this.slider(sub, 'Merge below (u/s)', () => nb.mergeValue, (v) => { nb.mergeValue = v; }, 0, MERGE_SPEED_MAX, 0.05, 2);
        const hot = this.checkbox(sub, 'Hot particles stick', () => nb.hotStickOn, (v) => { nb.hotStickOn = v; });
        this.slider(hot, 'Glowing: + (u/s)', () => nb.hotStickValue, (v) => { nb.hotStickValue = v; }, 0, HOT_MERGE_SPEED_MAX, 0.1, 1);
        sub = this.checkbox(sim, 'Breaking', () => nb.breakOn, (v) => { nb.breakOn = v; });
        this.slider(sub, 'Break above (u/s)', () => nb.breakValue, (v) => { nb.breakValue = v; }, 0, FRAGMENT_SPEED_MAX, 0.1, 1);

        this.constant(sim, 'Time scale (sim s / s)', CONSTANT_TIME_FACTOR, 0, 0.2, 0.001, 3);
        this.constant(sim, 'Gravity G', CONSTANT_GRAVITY, 0, 0.01, 0.0001, 4);
        this.constant(sim, 'Bounciness e', CONSTANT_RESTITUTION, 0, 1, 0.01, 2);
        this.constant(sim, 'Heat per energy', CONSTANT_HEAT_FACTOR, 0, 0.02, 0.0001, 4);
        this.constant(sim, 'Cooling time (s)', CONSTANT_COOL_SECONDS, 0.5, 10, 0.1, 1);
        this.button(sim, 'Defaults', () => {
            nb.resetSimulationConstants();
            this.refresh();
        });

        // --- Display ---------------------------------------------------------------------------
        const display = this.section('Display', true);
        this.checkbox(display, 'Planes (P)', () => nb.showPlanes, (v) => { nb.showPlanes = v; });
        this.checkbox(display, 'Axis gizmo', () => nb.showGizmo, (v) => { nb.showGizmo = v; });
        this.checkbox(display, 'Orbit when idle', () => nb.autoRotate, (v) => { nb.autoRotate = v; });
        this.checkbox(display, 'Lit spheres', () => nb.litSpheres, (v) => { nb.litSpheres = v; });
        this.checkbox(display, 'Spin marks', () => nb.spinMarks, (v) => { nb.spinMarks = v; });
        sub = this.checkbox(display, 'Trails (T)', () => nb.showTrails, (v) => { nb.showTrails = v; });
        this.slider(sub, 'Trail length (s)', () => nb.trailSeconds, (v) => { nb.trailSeconds = v; }, 0, TRAIL_SECONDS_MAX, 0.01, 2);
        sub = this.checkbox(display, 'Bloom', () => nb.bloomOn, (v) => { nb.bloomOn = v; });
        this.slider(sub, 'Bloom strength', () => nb.bloomStrength, (v) => { nb.bloomStrength = v; }, 0, 2, 0.01, 2);
        this.slider(sub, 'Bloom threshold', () => nb.bloomThreshold, (v) => { nb.bloomThreshold = v; }, 0, 2, 0.01, 2);

        // --- Units: what the sliders' units are -------------------------------------------------
        const units = this.section('Units', false);
        for (const line of [
            'u: scene unit; the planes\' grid lines are 2 u apart',
            's: second. Speeds are in u per simulated second;',
            '  the time scale is simulated seconds per real one;',
            '  trail length and cooling time are real seconds',
            'Masses follow the volume: 20 at radius 0.04 (312 at 0.1),',
            '  90000 at the attractors\' radius 0.5',
            'Gravity: a = G m r / (r^2 + s)^0.75 (not 1 / r^2), s = 0.05,',
            '  softened by size: at least the two\'s mean radius^2',
            'e: speed apart after a bounce over speed together before;',
            '  hard hits: e (above / impact speed)^(1/4)',
            'mu: friction impulse over the normal one, at most',
            'Merging: below the merge speed, plus as much as',
            '  \'Glowing: +\' times the pair\'s glow (1 - e^-heat)',
            'mu_r: rolling couple over contact radius times the',
            '  normal impulse, at most',
            'Heat rises by heat per energy times the kinetic',
            '  energy lost per unit of the particle\'s mass (u^2/s^2)',
            'Cooling time: the heat falls to 1/e in it',
            'Fragment room: free slots at the start, a share of',
            '  the particles placed',
        ]) {
            this.element('div', 'note', units, line);
        }

        // --- New simulation: applied on Restart -------------------------------------------------
        const fresh = this.section('New simulation', true);
        this.slider(fresh, 'Attractors', () => ui.attractors, (v) => { ui.attractors = Math.round(v); }, 1, MAX_ATTRACTORS, 1, 0);
        // The choices, and the one the URL asked for if it isn't among them.
        const choices = PARTICLE_CHOICES.slice();
        if (!choices.includes(ui.particles)) {
            choices.push(ui.particles);
            choices.sort((a, b) => a - b);
        }
        this.select(fresh, 'Particles each', choices.map(String),
            () => choices.indexOf(ui.particles), (i) => { ui.particles = choices[i]; });
        this.select(fresh, 'Attractor layout', ATTRACTOR_LAYOUTS, () => ui.attractorLayout, (i) => {
            ui.attractorLayout = i;
            this.refresh();
        });
        const radiusRow = this.slider(fresh, 'Layout radius (u)', () => ui.layoutRadius, (v) => { ui.layoutRadius = v; }, 1, LAYOUT_RADIUS_MAX, 0.1, 1);
        this.refreshers.push(() => { radiusRow.hidden = ui.attractorLayout === ATTRACTOR_LAYOUT_PRESET; });
        radiusRow.hidden = ui.attractorLayout === ATTRACTOR_LAYOUT_PRESET;
        this.slider(fresh, 'Attractor radius (u)', () => ui.heavyRadius, (v) => { ui.heavyRadius = v; }, 0.1, HEAVY_RADIUS_MAX, 0.01, 2);
        this.select(fresh, 'Particle layout', PARTICLE_LAYOUTS, () => ui.particleLayout, (i) => { ui.particleLayout = i; });
        this.slider(fresh, 'Particle radius min (u)', () => ui.lightRadiusMin, (v) => {
            ui.lightRadiusMin = v;
            if (ui.lightRadiusMax < v) {
                ui.lightRadiusMax = v;
            }
            this.refresh();
        }, LIGHT_RADIUS_SLIDER_MIN, LIGHT_RADIUS_SLIDER_MAX, 0.005, 3);
        this.slider(fresh, 'Particle radius max (u)', () => ui.lightRadiusMax, (v) => {
            ui.lightRadiusMax = v;
            if (v < ui.lightRadiusMin) {
                ui.lightRadiusMin = v;
            }
            this.refresh();
        }, LIGHT_RADIUS_SLIDER_MIN, LIGHT_RADIUS_SLIDER_MAX, 0.005, 3);
        this.slider(fresh, 'Cloud spread (u)', () => ui.spread, (v) => { ui.spread = v; }, 0.2, 3.0, 0.01, 2);
        this.slider(fresh, 'Fragment room (%)', () => 100.0 * ui.fragmentRoom, (v) => { ui.fragmentRoom = 0.01 * v; }, 0, 100, 1, 0);
        this.slider(fresh, 'Random seed', () => ui.seed, (v) => { ui.seed = Math.round(v); }, 1, 99999, 1, 0);
        this.button(fresh, 'Restart', () => nb.requestRestart());
        const running = this.element('div', 'note', fresh);
        const updateRunning = () => {
            running.textContent = nb.aliveCount === nb.numParticles
                ? `${nb.numParticles} particles running`
                : `${nb.aliveCount} particles running, in ${nb.numParticles} slots`;
        };
        this.refreshers.push(updateRunning);
        updateRunning();
        // The count changes as particles merge and break, not with a control.
        setInterval(updateRunning, 500);

        // The keys (Space, C, H, P, T) change the switches too.
        window.addEventListener('nbody-keys', () => this.refresh());

        this.refresh();
    }
}
