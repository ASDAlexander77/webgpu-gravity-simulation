class View
{
    constructor(canvasRef, adapter, device, context) {
        this.device = device;
        this.context = context;
        this.#configureCanvas(canvasRef, adapter, device);
    }

    get RenderPassDescriptor() {
        if (this.renderPassDescriptor)
        {
            this.updateRenderPassDescriptor();
            return this.renderPassDescriptor;
        }

        return this.renderPassDescriptor = this.createRenderPassDescriptor();
    }

    get View() {
        return this.context.getCurrentTexture().createView();
    }

    createRenderPassDescriptor() {
        const renderPassDescriptor = {
            colorAttachments: [
                {
                    view: this.View,
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        };

        return renderPassDescriptor;
    }

    updateRenderPassDescriptor()
    {
        this.renderPassDescriptor.colorAttachments[0].view = this.View;
    }

    #configureCanvas(canvasRef, adapter, device) {

        const context = this.context;

        const devicePixelRatio = window.devicePixelRatio || 1;
        const presentationSize = this.presentationSize = [
            canvasRef.clientWidth * devicePixelRatio,
            canvasRef.clientHeight * devicePixelRatio,
        ];
        const presentationFormat = this.presentationFormat = context.getPreferredFormat(adapter);

        context.configure({
            device,
            format: presentationFormat,
            compositingAlphaMode: 'premultiplied',
            //compositingAlphaMode: 'opaque',
            size: presentationSize,
        });
    }
}

class MSAAView extends View
{
    constructor(canvasRef, adapter, device, context) {
        super(canvasRef, adapter, device, context);
    }

    static MSAA = 4;

    get MultiSample() {
        return MSAAView.MSAA;
    }

    get View() {
        return this.viewTexture || (this.viewTexture = this.#createViewTexture());
    }

    get ResolveTarget() {
        return this.context.getCurrentTexture().createView();
    }

    #createViewTexture()
    {
        const texture = this.device.createTexture({
            size: this.presentationSize,
            sampleCount: MSAAView.MSAA,
            format: this.presentationFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        return texture.createView();
    }

    createRenderPassDescriptor() {
        const renderPassDescriptorMSAA = {
            colorAttachments: [
                {
                    view: this.View,
                    resolveTarget: this.ResolveTarget,
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'discard',
                },
            ],
        };

        return renderPassDescriptorMSAA;
    }

    updateRenderPassDescriptor()
    {
        this.renderPassDescriptor.colorAttachments[0].resolveTarget = this.ResolveTarget;
    }    
}