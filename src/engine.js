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

    compute(mesh, copy) {
        this.#compute(mesh, copy);
    }

    async #getWebGPU() {
        const canvasRef = this.canvasRef = document.getElementById("renderCanvas");
        const adapter = this.adapter = await navigator.gpu.requestAdapter();

        //const features = [];
        //for (const v of adapter.features.values()) features.push(v);

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
        this.view.EnableDepth = true;

        device.onuncapturederror = (event) => {
            if (event.error instanceof GPUValidationError) {
                console.error(`An uncaught WebGPU validation error was raised: ${event.error.message}`);
            }
        };
    }

    #createParams() {
        this.scale = 1.0;
        this.xangle = 0.0;
        this.yangle = 0.0;
        this.zangle = 0.0;

        this.paramsData = new UniformData(
            this,
            this.#getViewMatrix(),
        );        
    }

    #getViewMatrix() {
        const view = mat4.create();
        mat4.scale(view, view, vec3.fromValues(this.scale, this.scale, 1));

        if (this.view.EnableDepth)
        {
            mat4.scale(view, view, vec3.fromValues(2.5, 2.5, 1));
            mat4.translate(view, view, vec3.fromValues(0, 0, -4));

            mat4.rotateX(view, view, this.xangle);            
            mat4.rotateY(view, view, this.yangle);            

            const projectionMatrix = mat4.create();
            const aspect = 1;
            mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, aspect, 1, 100.0);

            let viewProj = mat4.create();
            mat4.multiply(viewProj, projectionMatrix, view);

            return viewProj;
        }

        return view;
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

        let vis_cnt = 0;
        const cut_vis_cnt = 50;
        const cnt = instancesCloneData.Data.length / 16;
        for (let i = 0; i < cnt; i++) {
            const offset = i * 16;

            if (data[12 + offset] <= 0.0)
                continue;

            vis_cnt++;
            if (vis_cnt > cut_vis_cnt)
            {
                break;
            }

            /*
            r += `<tr><td>F</td><td>${data[0 + offset].toFixed(12)}</td><td>${data[1 + offset].toFixed(12)}</td><td>${data[2 + offset].toFixed(12)}</td></tr>
                  <tr><td>Pos</td><td>${data[4 + offset].toFixed(12)}</td><td>${data[5 + offset].toFixed(12)}</td><td>${data[6 + offset].toFixed(12)}</td></tr>
                  <tr><td>V</td><td>${data[8 + offset].toFixed(12)}</td><td>${data[9 + offset].toFixed(12)}</td><td>${data[10 + offset].toFixed(12)}</td></tr>
                  <tr><td>Mass</td><td>${data[12 + offset].toFixed(12)}</td><td>Size</td><td>${data[13 + offset].toFixed(12)}</td></tr>
                  <tr><td>DBG1: R</td><td>${data[14 + offset].toFixed(12)}</td><td>DBG2: Dist</td><td>${data[15 + offset].toFixed(12)}</td></tr>`;
            */
            r += `<tr><td>Pos</td><td>${data[4 + offset].toFixed(12)}</td><td>${data[5 + offset].toFixed(12)}</td><td>${data[6 + offset].toFixed(12)}</td></tr>`;
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
            
            this.paramsData.Update(this.#getViewMatrix());
        }

        this.canvasRef.onmousedown = (evt) => {
            this.mousePressed = true;
        }

        this.canvasRef.onmouseup = (evt) => {
            this.mousePressed = false;
        }

        this.canvasRef.onmousemove = (evt) => {

            if (!this.mousePressed || evt.movementX == 0 && evt.movementY == 0)
            {
                return;
            }

            evt.preventDefault();

            this.xangle += evt.movementY * 0.001;            
            this.yangle += evt.movementX * 0.001;            
            
            this.paramsData.Update(this.#getViewMatrix());
        }        
    }

    async #drawLogic() {
        const device = this.device;

        const renderPassDescriptor = this.view.RenderPassDescriptor;

        const commandEncoders = [];
        const commandEncoder = device.createCommandEncoder();

        for (const mesh of this.scene.meshes) {

            mesh.Swap();

            if (mesh.computeShaderModule)
            {
                // calc
                const passEncoder = commandEncoder.beginComputePass();
                passEncoder.setPipeline(mesh.ComputePipeline);
                passEncoder.setBindGroup(0, mesh.CurrentBindGroup());
                passEncoder.setBindGroup(1, mesh.ParamsBindGroup);
                passEncoder.dispatchWorkgroups(Math.ceil(mesh.instanceCount / 64));
                passEncoder.end();
            }

            if (mesh.shaderModule)
            {
                const drawBuffer = mesh.CurrentInstancesData();

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
            }

            if (DEBUG) {
                const copyCmdDbg = mesh.GetCopyDataToCloneCmd();
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

    #compute(mesh, copy) {
        const device = this.device;

        mesh.Swap();

        const commandEncoders = [];
        const commandEncoder = device.createCommandEncoder();

        if (mesh.computeShaderModule)
        {
            // calc
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(mesh.ComputePipeline);
            passEncoder.setBindGroup(0, mesh.CurrentBindGroup());
            passEncoder.setBindGroup(1, mesh.ParamsBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(mesh.instanceCount / 64));
            passEncoder.end();

            if (copy) {
                const copyCmdDbg = mesh.GetCopyDataToCloneCmd();
                commandEncoders.push(copyCmdDbg.finish());
            }
        }

        commandEncoders.unshift(commandEncoder.finish());
        device.queue.submit(commandEncoders);

        return true;
    }    
}