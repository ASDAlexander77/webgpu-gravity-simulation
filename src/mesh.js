class Mesh {
    constructor(engine) {
        this.engine = engine;
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

    CurrentInstancesData(tick) {
        if (this.InstancesSwapDataBuffer) {
            return tick % 2
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

    GetCopyDataToCloneCmd(tick)
    {
        const drawBuffer = this.CurrentInstancesData(tick);

        const copyBufferCmd = this.engine.device.createCommandEncoder();

        copyBufferCmd.copyBufferToBuffer(
            drawBuffer,
            0,
            this.InstancesCloneDataBuffer,
            0,
            this.instancesData.data.byteLength);

        return copyBufferCmd;
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

    CurrentBindGroup(tick) {
        return tick % 2
            ? this.InstancesDataBindGroup
            : this.InstancesSwapDataBindGroup;
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

class SquaresMesh extends ComputedMesh {
    constructor(engine, count) {
        super(engine);
        this.vertexCount = 4;
        this.instanceCount = count;

        const z = 0.0;
        const radius = 1;
        const _size = 0.01;
        const diam = 2 * radius;

        const instancesInitData = [];
        for (let i = 0; i < count; i++) {
            // force
            instancesInitData.push(0);
            instancesInitData.push(0);
            instancesInitData.push(0);
            instancesInitData.push(0); // align, FYI: without align compute shader will not point to correct offset

            // position
            instancesInitData.push(diam * Math.random() - radius);
            instancesInitData.push(diam * Math.random() - radius);
            instancesInitData.push(z);
            instancesInitData.push(0); // align, FYI: without align compute shader will not point to correct offset

            // velocity
            instancesInitData.push(0);
            instancesInitData.push(0);
            instancesInitData.push(0);
            instancesInitData.push(0); // align, FYI: without align compute shader will not point to correct offset

            // mass, size, align, align
            const size = _size * Math.random() + _size / 10;            
            const mass = size;
            instancesInitData.push(mass);
            instancesInitData.push(size);
            instancesInitData.push(0); // align, FYI: without align compute shader will not point to correct offset
            instancesInitData.push(0); // align, FYI: without align compute shader will not point to correct offset
        }

        const instanceLayout = [
            // instance position
            { location: 0, type: 'float32', count: 4 },
            // instance position
            { location: 1, type: 'float32', count: 4 },
            // instance velocity
            { location: 2, type: 'float32', count: 4 },
            // mass, size
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

        const paramsInitData = [
            // delta time
            50.0,
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
}