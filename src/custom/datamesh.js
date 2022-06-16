
class DataMesh extends ComputedMesh {
    constructor(engine, count) {
        super(engine);
        this.instanceCount = count;

        const instancesInitData = [];
        for (let i = 0; i < count; i++) {
            // position
            instancesInitData.push(1);
            instancesInitData.push(2);
            instancesInitData.push(3);
            instancesInitData.push(0); // align, FYI: without align compute shader will not point to correct offset

            // data
            instancesInitData.push(4);
            instancesInitData.push(5);
            instancesInitData.push(6); // align, FYI: without align compute shader will not point to correct offset
            instancesInitData.push(0); // align, FYI: without align compute shader will not point to correct offset
        }

        const instanceLayout = [
            // instance position
            { location: 0, type: 'float32', count: 4 },
            // data
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
            0,
            0,
            0,
        ];

        this.paramsData = new UniformData(
            engine,
            paramsInitData,
        );
   }
}