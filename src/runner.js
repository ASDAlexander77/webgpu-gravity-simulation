class Runner {
    constructor() {
        this.engine = new Engine();
    }

    async run() {
        await this.engine.init();
        await this.#runLogic();
    }

    async #runLogic() {
        // 300 000 - crash, 100 000 - very slow
        // 35 000 speed is fine       
        //await this.#createScene(350);
        await this.#createScene(350);
        this.engine.draw();
    }

    async #createScene(count) {
        const scene = new Scene();

        this.engine.setScene(scene);

        const gravityMesh = new GravityMesh(this.engine, count);
        gravityMesh.Shader = "draw_wgsl";
        gravityMesh.ComputeShader = "calc_wgsl";
        await gravityMesh.addSvg("svg1");

        scene.addMesh(gravityMesh);
    }
}