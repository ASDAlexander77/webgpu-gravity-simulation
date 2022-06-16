
class GravityMesh extends ComputedMesh {
    constructor(engine, count) {
        super(engine);
        this.vertexCount = 4;
        this.instanceCount = count;

        const z = 0.0;
        const radius = 1;
        const _size = 0.0075;
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
            //instancesInitData.push(radius * Math.random());
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