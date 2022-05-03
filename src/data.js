function getTypeSize(type) {
    switch (type) {
        case "float32": return 4;
        case "float64": return 8;
    }

    throw "No data"
}

class Data {
    constructor(engine, data) {
        this.engine = engine;
        this.data = new Float32Array(data.length)
        this.data.set(data);
    }

    get Data() {
        return this.data;
    }

    get Buffer() {
        return this.buffer || (this.buffer = this.#createBuffer());
    }

    get ResourceInfo() {
        return this.resourceInfo || (this.resourceInfo = this.createResourceInfo());
    }

    async LoadData() {
        try {
            await this.Buffer.mapAsync(GPUMapMode.READ);
            this.data.set(new Float32Array(this.Buffer.getMappedRange()));
        }
        finally {
            this.Buffer.unmap();
        }
    }

    Update(newValues) {
        this.engine.device.queue.writeBuffer(
            this.Buffer,
            0,
            new Float32Array(newValues)
        );
    }

    createResourceInfo() {
        const resource = {
            buffer: this.Buffer,
            //offset: 0,
            //size: this.data.byteLength,
        };

        return resource;
    }

    #createBuffer() {
        const buffer = this.engine.device.createBuffer({
            size: this.data.byteLength,
            usage: this.usage,
            mappedAtCreation: true,
        });

        new Float32Array(buffer.getMappedRange()).set(this.data);
        buffer.unmap();
        return buffer;
    }
}

class UniformData extends Data {
    constructor(engine, data) {
        super(engine, data);

        this.usage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    }

    getBindGroupFor(bindGroupLayout) {

        if (this.bindGroupLayout == bindGroupLayout)
        {
            return this.uniformBindGroup;
        }

        this.bindGroupLayout = bindGroupLayout;

        const uniformBindGroup = this.engine.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.ResourceInfo,
                },
            ],
        });

        return this.uniformBindGroup = uniformBindGroup;
    }
}

class InstanceData extends Data {
    constructor(engine, data, strides) {
        super(engine, data);

        this.attributes = [];
        this.stepMode = 'instance';
        this.usage = GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;

        // calculate attributes
        let offset = 0;
        for (const stride of strides) {
            this.#addAttribute(
                stride.location,
                offset,
                `${stride.type}x${stride.count}`,
            );

            offset += stride.count * getTypeSize(stride.type);
        }

        this.arrayStride = offset;
    }

    get BufferDescriptor() {
        const bufferDescr = {
            arrayStride: this.arrayStride,
            stepMode: this.stepMode,
            attributes: this.attributes,
        };

        return bufferDescr;
    }

    #addAttribute(shaderLocation, offset, format) {
        const attrDescr = {
            shaderLocation: shaderLocation,
            offset: offset,
            format: format,
        };

        this.attributes.push(attrDescr);
    };
}

class InstanceCloneData extends InstanceData {
    constructor(engine, data, strides) {
        super(engine, data, strides);

        this.usage = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;
    }
}

class VertexData extends InstanceData {
    constructor(engine, data, strides, topology) {
        super(engine, data, strides);
        this.stepMode = 'vertex';
        this.usage = GPUBufferUsage.VERTEX;
        this.topology = topology;
    }
}
