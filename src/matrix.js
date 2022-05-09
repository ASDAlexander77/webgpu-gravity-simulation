class Matrix
{
    constructor(dimX, dimY, data)
    {
        this.dimX = dimX;
        this.dimY = dimY;
        this.data = data;
    }

    get Buffer()
    {
        return this.data;
    }

    Scale(scale) 
    {
        let i = 0;
        for (const val of this.data)
        {
            this.data[i++] = val * scale;
        }
    }

    Multiply(otherMatrix)
    {
        throw "to do";
    }
}

class Matrix4 extends Matrix
{
    constructor(data)
    {
        super(4, 4, data);
    }

    static Zero()
    {
        return new Matrix4(
            [
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0
            ]);
    }

    static One()
    {
        return new Matrix4(
            [
                1.0, 0, 0, 0,
                0, 1.0, 0, 0,
                0, 0, 1.0, 0,
                0, 0, 0, 1.0
            ]);
    }

    Zoom(scale) 
    {
        this.data[15] = 1 / scale;
    }

    Perspective(angleOfView = 90, imageAspectRatio = 1, near = 0.1, far = 100) 
    {
        const scale = Math.tan(angleOfView * 0.5 * Math.PI / 180) * near; 
        const r = imageAspectRatio * scale;
        const l = -r; 
        const t = scale;
        const b = -t;      
        
        const data = this.data;

        data[0] = 2 * near / (r - l); 
        data[1] = 0; 
        data[2] = 0; 
        data[3] = 0; 
     
        data[4] = 0; 
        data[5] = 2 * near / (t - b); 
        data[6] = 0; 
        data[7] = 0; 
     
        data[8] = (r + l) / (r - l); 
        data[9] = (t + b) / (t - b); 
        data[10] = -(far + near) / (far - near); 
        data[11] = -1; 
     
        data[12] = 0; 
        data[13] = 0; 
        data[14] = -2 * far * near / (far - near); 
        data[15] = 0;         
    }
}