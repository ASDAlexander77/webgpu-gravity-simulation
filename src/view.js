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

    get DepthTexture()
    {
        return this.depthTexture || (this.depthTexture = this.#createDepthTexture());
    }

    get EnableDepth() {
        return this.enableDepth;
    }

    set EnableDepth(value) {
        this.enableDepth = value;
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

        if (this.enableDepth)
        {
            renderPassDescriptor.depthStencilAttachment = this.#getDepthStencilAttachment();
        }

        return renderPassDescriptor;
    }

    updateRenderPassDescriptor()
    {
        this.renderPassDescriptor.colorAttachments[0].view = this.View;
    }

    getDepthTextureDescriptor()
    {
        const depthTextureDescriptor = {
            size: this.presentationSize,
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        }

        return depthTextureDescriptor;
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

    #getDepthStencilAttachment()
    {
        const depthStencilAttachment = {
            view: this.DepthTexture.createView(),
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        };

        return depthStencilAttachment;
    }

    #createDepthTexture()
    {
        return this.device.createTexture(this.getDepthTextureDescriptor());
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

    createRenderPassDescriptor() {
        const renderPassDescriptorMSAA = super.createRenderPassDescriptor();
        renderPassDescriptorMSAA.resolveTarget = this.ResolveTarget;
        renderPassDescriptorMSAA.storeOp = 'discard';
        return renderPassDescriptorMSAA;
    }

    updateRenderPassDescriptor()
    {
        this.renderPassDescriptor.colorAttachments[0].resolveTarget = this.ResolveTarget;
    }    

    getDepthTextureDescriptor()
    {
        const depthTextureDescriptor = super.getDepthTextureDescriptor();
        depthTextureDescriptor.sampleCount = MSAAView.MSAA;
        return depthTextureDescriptor;
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
}