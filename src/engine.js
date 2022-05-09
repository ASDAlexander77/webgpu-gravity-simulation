//const DEBUG = true;
const DEBUG = false;

class Engine {
    constructor() {
        this.meshes = [];
        this.tick = 0;
    }

    async init() {
        await this.#getWebGPU();
        this.#createView();
        this.#createParams();
        this.#attachToEvents();
    }

    setScene(scene) {
        this.scene = scene;
    }

    draw() {
        this.#draw();
    }

    async #getWebGPU() {
        const canvasRef = this.canvasRef = document.getElementById("renderCanvas");
        const adapter = this.adapter = await navigator.gpu.requestAdapter();
        this.device = await adapter.requestDevice();
        this.context = canvasRef.getContext('webgpu');
    }

    #createView() {
        const canvasRef = this.canvasRef;
        const adapter = this.adapter;
        const device = this.device;
        const context = this.context;

        //this.view = new View(canvasRef, adapter, device, context);
        this.view = new MSAAView(canvasRef, adapter, device, context);
        //this.view.EnableDepth = true;

        device.onuncapturederror = (event) => {
            if (event.error instanceof GPUValidationError) {
                console.error(`An uncaught WebGPU validation error was raised: ${event.error.message}`);
            }
        };
    }

    #createParams() {
        this.scale = 1.0;

        this.paramsData = new UniformData(
            this,
            this.#getViewMatrix().Buffer,
        );        
    }

    #getViewMatrix() {
        const viewMatrix = Matrix4.Create();
        viewMatrix.Scale(this.scale);
        viewMatrix.data[15] = 1.0; // needed for correct scaling
        return viewMatrix;
    }

    #draw() {
        const self = this;

        async function frame() {
            const r = await self.#drawLogic();
            if (r) {
                if (DEBUG) {
                    requestAnimationFrame(frame);
                    //setTimeout(frame, 100);
                }
                else {
                    requestAnimationFrame(frame);
                }
            }
        }

        requestAnimationFrame(frame);
    }

    #validationStart() {
        this.device.pushErrorScope('validation');
    }

    #outOfMemoryStart() {
        this.device.pushErrorScope('out-of-memory');
    }

    #errorStop() {
        this.device.popErrorScope().then((error) => {
            if (error) {
                console.error(`An error occured while creating sampler: ${error.message}`);
            }
        });
    }

    async #printDebugInfo(instancesCloneData) {
        await instancesCloneData.LoadData();
        const data = instancesCloneData.Data;

        let r = `<table id="tbldbg"><tr><td></td><td>x</td><td>y</td><td>z</td></tr>`;

        const cnt = Math.min(instancesCloneData.Data.length / 16, 10);
        for (let i = 0; i < cnt; i++) {
            const offset = i * 16;
            r += `<tr><td>F</td><td>${data[0 + offset].toFixed(12)}</td><td>${data[1 + offset].toFixed(12)}</td><td>${data[2 + offset].toFixed(12)}</td></tr>
                  <tr><td>Pos</td><td>${data[4 + offset].toFixed(12)}</td><td>${data[5 + offset].toFixed(12)}</td><td>${data[6 + offset].toFixed(12)}</td></tr>
                  <tr><td>V</td><td>${data[8 + offset].toFixed(12)}</td><td>${data[9 + offset].toFixed(12)}</td><td>${data[10 + offset].toFixed(12)}</td></tr>
                  <tr><td>Mass</td><td>${data[12 + offset].toFixed(12)}</td><td>Size</td><td>${data[13 + offset].toFixed(12)}</td></tr>
                  <tr><td>DBG1: R</td><td>${data[14 + offset].toFixed(12)}</td><td>DBG2: Dist</td><td>${data[15 + offset].toFixed(12)}</td></tr>`;
        }

        r += `</table>`;

        document.getElementById("DEBUG").innerHTML = r;

        //console.log(data);        
    }

    #attachToEvents()
    {
        this.canvasRef.onwheel = (evt) => {
            evt.preventDefault();

            this.scale += evt.deltaY * -0.0001;
            
            this.paramsData.Update(this.#getViewMatrix().Buffer);
        }
    }

    async #drawLogic() {
        const device = this.device;

        this.tick++;

        const renderPassDescriptor = this.view.RenderPassDescriptor;

        const commandEncoders = [];
        const commandEncoder = device.createCommandEncoder();

        for (const mesh of this.scene.meshes) {

            {
                // calc
                const passEncoder = commandEncoder.beginComputePass();
                passEncoder.setPipeline(mesh.ComputePipeline);
                passEncoder.setBindGroup(0, mesh.CurrentBindGroup(this.tick));
                passEncoder.setBindGroup(1, mesh.ParamsBindGroup);
                passEncoder.dispatchWorkgroups(Math.ceil(mesh.instanceCount / 64));
                passEncoder.end();
            }

            const drawBuffer = mesh.CurrentInstancesData(this.tick);

            // draw
            {
                //this.#validati4onStart();

                const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

                const pipeline = mesh.getRenderPipeline(this.view);

                passEncoder.setPipeline(pipeline);

                if (drawBuffer) {
                    passEncoder.setVertexBuffer(0, drawBuffer);
                }

                if (mesh.VertexBuffer) {
                    passEncoder.setVertexBuffer(1, mesh.VertexBuffer);
                }

                if (mesh.HasTexture) {
                    passEncoder.setBindGroup(0, mesh.TextureBindGroup);
                }

                passEncoder.setBindGroup(1, this.paramsData.getBindGroupFor(pipeline.getBindGroupLayout(1)));

                const firstVertex = 0;
                const firstInstance = 0;
                passEncoder.draw(mesh.vertexCount, mesh.instanceCount, firstVertex, firstInstance);
                passEncoder.end();

                //this.#errorStop();
            }

            if (DEBUG) {
                const copyCmdDbg = mesh.GetCopyDataToCloneCmd(this.tick);
                commandEncoders.push(copyCmdDbg.finish());
            }
        }

        commandEncoders.unshift(commandEncoder.finish());
        device.queue.submit(commandEncoders);

        if (DEBUG) {
            for (const mesh of this.scene.meshes) {
                await this.#printDebugInfo(mesh.instancesCloneData);
            }
        }

        return true;
    }
}