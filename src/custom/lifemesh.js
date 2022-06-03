
class LifeMesh extends ComputedMesh {
    constructor(engine, width) {
        super(engine);
        this.vertexCount = 4;
        this.instanceCount = width * width;

        const z = 0.0;
        const radius = 1;

        const step = (2 * radius) / (width - 1);

        const instancesInitData = [];
        for (let y = 0; y < width; y++)
            for (let x = 0; x < width; x++) {
                // position
                instancesInitData.push(x * step - radius);
                instancesInitData.push(y * step - radius);
                instancesInitData.push(z);
                //instancesInitData.push(radius * Math.random());
                instancesInitData.push(0); // align, FYI: without align compute shader will not point to correct offset

                // size, align, align, align
                let size = step;            

                if (Math.random() < 0.5)
                {
                    size = 0;
                }

                instancesInitData.push(size);
                instancesInitData.push(0);
                instancesInitData.push(0); // align, FYI: without align compute shader will not point to correct offset
                instancesInitData.push(0); // align, FYI: without align compute shader will not point to correct offset
            }

        const instanceLayout = [
            // instance position
            { location: 0, type: 'float32', count: 4 },
            // size, align, align
            { location: 1, type: 'float32', count: 4 },
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
            width,
            step,
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