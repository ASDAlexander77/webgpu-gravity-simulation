class Mesh {
    constructor(engine) {
        this.engine = engine;
        this.tick = -1;
    }

    set Shader(elementId) {
        const shaderText = document.getElementById(elementId).value;
        const codeDescr = { code: shaderText };
        this.shaderModule = this.engine.device.createShaderModule(codeDescr);
    }

    get Sampler() {
        return this.sampler || (this.sampler = this.#createSampler());
    }

    get HasTexture() {
        return !!this.texture;
    }

    get InstancesDataBuffer() {
        return this.instancesData.Buffer;
    }

    get InstancesSwapDataBuffer() {
        return this.instancesSwapData.Buffer;
    }

    Swap() {
        this.tick++;
    }

    CurrentInstancesData() {
        if (this.InstancesSwapDataBuffer) {
            return this.tick % 2
                ? this.InstancesSwapDataBuffer
                : this.InstancesDataBuffer;
        }

        return this.InstancesDataBuffer;
    }

    get VertexBuffer() {
        return this.vertexData.Buffer;
    }

    get TextureBindGroup() {
        return this.textureBindGroup || (this.textureBindGroup = this.#createTextureBindGroup());
    }

    GetComputeAndDrawCmd(view, paramsData) {
        const commandEncoder = this.engine.device.createCommandEncoder();
        this.getPassesForComputeAndDrawCmd(commandEncoder, view, paramsData);
        return commandEncoder.finish();
    }

    getPassesForComputeAndDrawCmd(commandEncoder, view, paramsData) {
        if (this.shaderModule) {
            const drawBuffer = this.CurrentInstancesData();

            // draw
            {
                //this.#validationStart();
                const passEncoder = commandEncoder.beginRenderPass(view.RenderPassDescriptor);

                const pipeline = this.getRenderPipeline(view);

                passEncoder.setPipeline(pipeline);

                if (drawBuffer) {
                    passEncoder.setVertexBuffer(0, drawBuffer);
                }

                if (this.VertexBuffer) {
                    passEncoder.setVertexBuffer(1, this.VertexBuffer);
                }

                if (this.HasTexture) {
                    passEncoder.setBindGroup(0, this.TextureBindGroup);
                }

                passEncoder.setBindGroup(1, paramsData.getBindGroupFor(pipeline.getBindGroupLayout(1)));

                const firstVertex = 0;
                const firstInstance = 0;
                passEncoder.draw(this.vertexCount, this.instanceCount, firstVertex, firstInstance);
                passEncoder.end();

                //this.#errorStop();
            }
        }

        return commandEncoder;
    }

    GetCopyDataToCloneCmd() {
        const drawBuffer = this.CurrentInstancesData();

        const copyBufferCmd = this.engine.device.createCommandEncoder();

        copyBufferCmd.copyBufferToBuffer(
            drawBuffer,
            0,
            this.InstancesCloneDataBuffer,
            0,
            this.instancesData.data.byteLength);

        return copyBufferCmd.finish();
    }

    async addImage(imgId) {
        // load texture
        const img = document.getElementById(imgId);
        await img.decode();
        const imageBitmap = await createImageBitmap(img);
        this.#addImageAsTexture(imageBitmap);
    }

    async addSvg(svgId) {
        // load texture
        const svg = document.getElementById(svgId);
        const img = await this.#createImageBitmapFromSvg(svg);
        const imageBitmap = await createImageBitmap(img);
        this.#addImageAsTexture(imageBitmap);
    }

    getRenderPipeline(view) {
        if (this.renderPipelineParam != view.MultiSample) {
            this.renderPipeline = undefined;
        }

        this.renderPipelineParam = view.MultiSample;
        return this.renderPipeline || (this.renderPipeline = this.#createRenderPipeline(view));
    }

    #createRenderPipeline(view) {
        const pipelineDescr = {
            vertex: this.#getVertexDescr(),
            fragment: this.#getFragmentDescr(view),
            primitive: this.#getPrimitiveDescr(),
            layout: 'auto'
        };

        if (view.MultiSample) {
            pipelineDescr.multisample = {
                count: view.MultiSample,
            };
        }

        if (view.EnableDepth) {
            pipelineDescr.depthStencil = {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            };

            pipelineDescr.primitive.cullMode = 'back';
            // must have feature - "depth-clip-control"
            //pipelineDescr.primitive.unclippedDepth = true;
        }

        return this.engine.device.createRenderPipeline(pipelineDescr);
    }

    #getVertexDescr() {
        const vertexDescr = {
            module: this.shaderModule,
            entryPoint: 'vert_main',
            buffers: this.#getBuffersDescr()
        }

        return vertexDescr;
    }

    #getFragmentDescr(view) {
        const fragmentDescr = {
            module: this.shaderModule,
            entryPoint: 'frag_main',
            targets: [
                {
                    format: view.presentationFormat,
                },
            ],
        };

        return fragmentDescr;
    }

    #getBuffersDescr() {
        const buffersDescr = [];

        const instancesDataBuffer = this.#getInstancesDataBufferDescr();
        if (instancesDataBuffer) {
            buffersDescr.push(instancesDataBuffer);
        }

        const vertBuffer = this.#getVertexBufferDescr();
        if (vertBuffer) {
            buffersDescr.push(vertBuffer);
        }

        return buffersDescr;
    }

    #getInstancesDataBufferDescr() {
        return this.instancesData.BufferDescriptor;
    }

    #getVertexBufferDescr() {
        return this.vertexData.BufferDescriptor;
    }

    #getPrimitiveDescr() {
        const primitiveDescr = {
            topology: this.vertexData.topology,
        };

        return primitiveDescr;
    }

    #createTextureBindGroup() {
        if (!this.texture) {
            return;
        }

        const textureBindGroup = this.engine.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: this.Sampler,
                },
                {
                    binding: 1,
                    resource: this.texture.createView(),
                },
            ],
        });

        return textureBindGroup;
    }

    #createSampler() {
        const sampler = this.engine.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        return sampler;
    }


    async #createImageBitmapFromSvg(svgElement) {
        return new Promise(async (resolve) => {
            var xml = new XMLSerializer().serializeToString(svgElement);

            const svgBlob = new Blob([xml], {
                type: "image/svg+xml;charset=utf-8"
            });

            const svgBase64 = URL.createObjectURL(svgBlob);
            const img = new Image();
            img.onload = () => {
                resolve(img);
            }

            img.src = svgBase64;
        });
    }

    #addImageAsTexture(imageBitmap) {
        // create texture
        const texture = this.engine.device.createTexture({
            size: [imageBitmap.width, imageBitmap.height, 1],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.texture = texture;

        this.engine.device.queue.copyExternalImageToTexture(
            { source: imageBitmap },
            { texture: texture },
            [imageBitmap.width, imageBitmap.height]
        );
    }
}

class ComputedMesh extends Mesh {
    constructor(engine) {
        super(engine);
    }

    set ComputeShader(elementId) {
        const computeShaderText = document.getElementById(elementId).value;
        const codeDescr = { code: computeShaderText };
        this.computeShaderModule = this.engine.device.createShaderModule(codeDescr);
    }

    get ComputePipeline() {
        return this.computePipeline || (this.computePipeline = this.#createComputePipeline());
    }

    get InstancesDataBindGroup() {
        return this.instancesDataBindGroup || (this.instancesDataBindGroup = this.#createInstancesDataBindGroup());
    }

    get InstancesSwapDataBindGroup() {
        return this.instancesSwapDataBindGroup || (this.instancesSwapDataBindGroup = this.#createInstancesSwapDataBindGroup());
    }

    get ParamsBindGroup() {
        return this.paramsBindGroup || (this.paramsBindGroup = this.#createParamsBindGroup());
    }

    get ParamsBuffer() {
        return this.paramsData.Buffer;
    }

    get InstancesCloneDataBuffer() {
        return this.instancesCloneData.Buffer;
    }

    CurrentBindGroup() {
        return this.tick % 2
            ? this.InstancesDataBindGroup
            : this.InstancesSwapDataBindGroup;
    }

    getPassesForComputeAndDrawCmd(commandEncoder, view, paramsData) {
        if (this.computeShaderModule) {
            // calc
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.ComputePipeline);
            passEncoder.setBindGroup(0, this.CurrentBindGroup());
            passEncoder.setBindGroup(1, this.ParamsBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.instanceCount / 64));
            passEncoder.end();
        }

        super.getPassesForComputeAndDrawCmd(commandEncoder, view, paramsData);
    }

    #createComputePipeline() {
        const computePipelineDescr = {
            compute: {
                module: this.computeShaderModule,
                entryPoint: 'main',
            },
            layout: 'auto'
        };

        return this.engine.device.createComputePipeline(computePipelineDescr);
    }

    #createInstancesDataBindGroup() {

        const entries = [
            {
                binding: 0,
                resource: this.instancesData.ResourceInfo,
            },
            {
                binding: 1,
                resource: this.instancesSwapData.ResourceInfo,
            },
        ];

        const bindGroup = this.engine.device.createBindGroup({
            layout: this.ComputePipeline.getBindGroupLayout(0),
            entries,
        });

        return bindGroup;
    }

    #createInstancesSwapDataBindGroup() {

        const entries = [
            {
                binding: 0,
                resource: this.instancesSwapData.ResourceInfo,
            },
            {
                binding: 1,
                resource: this.instancesData.ResourceInfo,
            },
        ];

        const bindGroup = this.engine.device.createBindGroup({
            layout: this.ComputePipeline.getBindGroupLayout(0),
            entries,
        });

        return bindGroup;
    }

    #createParamsBindGroup() {
        const bindGroup = this.engine.device.createBindGroup({
            layout: this.ComputePipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: this.paramsData.ResourceInfo,
                },
            ],
        });

        return bindGroup;
    }
}
