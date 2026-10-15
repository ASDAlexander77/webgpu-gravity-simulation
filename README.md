# WebGPU N-Body Gravity Simulation

**A real-time 3D N-body gravity simulation with particle collisions, running entirely on the GPU using WebGPU compute shaders (WGSL) in the browser.** No build step, no dependencies: plain JavaScript and HTML.

[![WebGPU N-body gravity simulation with colliding particles](example.jpg)](index.html)

<p align="center">
  <a href="https://www.w3.org/TR/webgpu/" title="WebGPU"><img src="https://user-images.githubusercontent.com/454184/116650388-d2cdde00-a935-11eb-8553-eb73eed2016b.png" alt="WebGPU logo" width="150" height="150" /></a>
</p>

## Features

- **GPU-accelerated N-body physics**: every particle is attracted by every other particle (all-pairs gravity), computed in WebGPU compute shaders.
- **Collision physics**: particles are spheres that bounce off each other (restitution, positional correction) instead of merging, with a flash effect on impact.
- **Heavy attractors with particle clouds**: six massive centers, each surrounded by a cloud of light particles (1536 particles by default).
- **Instanced rendering**: particles are drawn as textured billboards (an SVG sprite) with a WebGPU render pipeline and multisampling.
- **Interactive camera**: drag with the mouse to rotate the view, use the mouse wheel to zoom.
- **Multi-pass compute pipeline** with ping-pong (double-buffered) storage buffers: `calculate_cs` → `count_cs` → `collide_cs` → `integrate_cs`.
- **Extra demos**: Conway's Game of Life on the GPU and a minimal WebGPU "hello triangle".

## Demos

| File | What it is |
| --- | --- |
| [`index.html`](index.html) | N-body gravity simulation with collisions (main demo) |
| [`index.life.html`](index.life.html) | Conway's Game of Life as a compute shader |
| [`index.hello.html`](index.hello.html) | Minimal WebGPU example: a red triangle |

## Requirements

- A browser with [WebGPU](https://caniuse.com/webgpu) support, such as a recent Chrome or Edge (113+), or Firefox / Safari versions that ship WebGPU.
- A GPU and drivers that support WebGPU.

## Quick start

1. Clone the repository:

   ```bash
   git clone https://github.com/ASDAlexander77/webgpu-gravity-simulation.git
   cd webgpu-gravity-simulation
   ```

2. Open `index.html` in a WebGPU-enabled browser.

   Opening the file directly works, but if your browser blocks WebGPU on `file://` pages, serve the folder over `localhost` instead:

   ```bash
   npx serve .
   # or
   python -m http.server 8000
   ```

   On Windows you can also launch Chrome with WebGPU enabled using the bundled script, [`webgpu.bat`](webgpu.bat):

   ```cmd
   "%ProgramFiles%\Google\Chrome\Application\chrome.exe" --enable-unsafe-webgpu file:///path/to/index.html
   ```

Controls:

| Input | Action |
| --- | --- |
| Left mouse drag | Rotate the camera |
| Mouse wheel | Zoom in / out |

## How it works

The simulation keeps all particle state (position, velocity, mass, radius) in GPU storage buffers and never reads it back to the CPU. Each frame runs four compute passes, then renders:

1. **`calculate_cs`**: sums the gravitational pull of all other particles and updates velocities.
2. **`count_cs`**: counts the contacts each particle has this step.
3. **`collide_cs`**: resolves sphere–sphere collisions (velocity change, overlap separation, impact flash).
4. **`integrate_cs`**: moves the particles.

The result is drawn as instanced billboards. The O(N²) all-pairs interaction is what makes the GPU worthwhile here: with a few thousand particles, every particle is tested against every other one several times per step.

Tunable constants (`GRAVITY`, `SOFTEN`, `TIME_FACTOR`, `RESTITUTION`, `SEPARATION`, …) are at the top of the compute shader in `index.html`; the particle count is set in `Runner.#createScene` in the same file.

## Project structure

```
index.html            N-body gravity demo (shaders + entry point)
index.life.html       Game of Life demo
index.hello.html      Minimal WebGPU triangle
src/
  engine.js           WebGPU device, canvas, render loop, camera input
  mesh.js             Mesh / computed mesh, pipelines, textures
  data.js             Vertex, instance and uniform buffer wrappers
  view.js, scene.js   View settings and scene container
  vector.js, matrix.js  Math helpers
  custom/
    gravitymesh.js    N-body particles with collisions
    lifemesh.js       Game of Life grid
    datamesh.js       Data-driven mesh example
```

## Performance notes

- Because every particle is compared with every other, cost grows quadratically with particle count. 1,536 particles run smoothly; tens of thousands become slow and hundreds of thousands can crash the GPU process.
- Spatial partitioning (grid or tree) would be the next step to scale further.

## Minimal WebGPU example

[`index.hello.html`](index.hello.html) shows the smallest useful WebGPU program: request an adapter and device, configure a canvas, create a render pipeline from WGSL vertex and fragment shaders, and draw a triangle.

[![WebGPU hello triangle example](example.gif)](index.hello.html)

## Related topics

WebGPU, WGSL, compute shaders, GPGPU, N-body simulation, gravity simulation, particle system, collision detection, physics simulation, JavaScript, real-time 3D graphics, Game of Life.
