# WebGPU Gravity Simulation (Compute)
###### Powered by [![WebGPU](https://user-images.githubusercontent.com/454184/116650388-d2cdde00-a935-11eb-8553-eb73eed2016b.png =300x300)](https://www.w3.org/TR/webgpu/)

# Example

```HTML
<html>
<!DOCTYPE html>

<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />

    <title></title>

    <style>
        html,
        body {
            overflow: hidden;
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
        }

        #renderCanvas {
            width: 800;
            height: 600;
            background-color: cadetblue;
            border: 1px red solid;
        }

        textarea {
            display: none;
        }
    </style>
</head>

<body>
    <textarea id="vert_code">
        @vertex
        fn main(@builtin(vertex_index) VertexIndex : u32)
             -> @builtin(position) vec4<f32> {
          var pos = array<vec2<f32>, 3>(
              vec2<f32>(0.0, 0.5),
              vec2<f32>(-0.5, -0.5),
              vec2<f32>(0.5, -0.5));
        
          return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
        }
    </textarea>
    <textarea id="frag_code">
        @fragment
        fn main() -> @location(0) vec4<f32> {
          return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
    </textarea>
    <canvas id="renderCanvas"></canvas>
    <script>
        async function init() {
            const canvasRef = document.getElementById("renderCanvas");
            const triangleVertWGSL = document.getElementById("vert_code").value;
            const redFragWGSL = document.getElementById("frag_code").value;

            const adapter = await navigator.gpu.requestAdapter();
            const device = await adapter.requestDevice();

            if (canvasRef === null) return;
            const context = canvasRef.getContext('webgpu');

            const devicePixelRatio = window.devicePixelRatio || 1;
            canvasRef.width = canvasRef.clientWidth * devicePixelRatio;
            canvasRef.height = canvasRef.clientHeight * devicePixelRatio;
            const presentationFormat = navigator.gpu.getPreferredCanvasFormat(adapter);

            context.configure({
                device,
                format: presentationFormat,
                alphaMode: 'premultiplied'
            });

            const pipeline = device.createRenderPipeline({
                vertex: {
                    module: device.createShaderModule({
                        code: triangleVertWGSL,
                    }),
                    entryPoint: 'main',
                },
                fragment: {
                    module: device.createShaderModule({
                        code: redFragWGSL,
                    }),
                    entryPoint: 'main',
                    targets: [
                        {
                            format: presentationFormat,
                        },
                    ],
                },
                primitive: {
                    topology: 'triangle-list',
                },
                layout: 'auto'
            });

            function frame() {
                if (!canvasRef) return;

                const commandEncoder = device.createCommandEncoder();
                const textureView = context.getCurrentTexture().createView();

                const renderPassDescriptor = {
                    colorAttachments: [
                        {
                            view: textureView,
                            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                            loadOp: 'clear',
                            storeOp: 'store',
                        },
                    ],
                };

                const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
                passEncoder.setPipeline(pipeline);
                passEncoder.draw(3, 1, 0, 0);
                passEncoder.end();

                device.queue.submit([commandEncoder.finish()]);
                requestAnimationFrame(frame);
            }

            requestAnimationFrame(frame);
        };

        window.addEventListener("DOMContentLoaded", init);
    </script>
</body>

</html>
```

Run
```cmd
"%ProgramFiles%\Google\Chrome\Application\chrome.exe" --enable-unsafe-webgpu file:///example.html
```

Result
[![Example](https://raw.githubusercontent.com/ASDAlexander77/webgpu-gravity-simulation/main/example.gif)](https://github.com/ASDAlexander77/webgpu-gravity-simulation)

