class Matrix
{
    constructor(rowsCount, columnsCount, data)
    {
        this.rowsCount = rowsCount;
        this.columnsCount = columnsCount;
        this.data = data;
    }

    get Buffer()
    {
        return this.data;
    }

    GetIndex(row, column)
    {
        return row * this.rowsCount + column;
    }

    GetByIndex(row, column)
    {
        return this.data[this.GetIndex(row, column)];
    }

    SetByIndex(row, column, value)
    {
        this.data[this.GetIndex(row, column)] = value;
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
        const m = this.rowsCount;
        const n = otherMatrix.rowsCount;
        const p = otherMatrix.columnsCount;

        const result = new Matrix(m, p, new Array(m * p));

        for (let i = 0; i < m; i++)
            for (let j = 0; j < p; j++)
                {
                    let sum = 0;
                    for (let k = 0; k < n; k++)
                        sum += this.GetByIndex(i, k) * otherMatrix.GetByIndex(k, j);
                    result.SetByIndex(i, j, sum);
                }

        // save
        return result;
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

    static RotateX(angle)
    {
        return new Matrix4(
            [
                1.0, 0,               0,                0,
                0,   Math.cos(angle), -Math.sin(angle), 0,
                0,   Math.sin(angle), Math.cos(angle),  0,
                0,   0,               0,                1.0
            ]);
    }    

    static RotateY(angle)
    {
        return new Matrix4(
            [
                Math.cos(angle),  0,   Math.sin(angle), 0,
                0,                1.0, 0,               0,
                -Math.sin(angle), 0,   Math.cos(angle), 0,
                0,                0,   0,               1.0
            ]);
    } 

    static RotateZ(angle)
    {
        return new Matrix4(
            [
                Math.cos(angle), -Math.sin(angle), 0,   0,
                Math.sin(angle), Math.cos(angle),  0,   0,
                0,               0,                1.0, 0,
                0,               0,                0,   1.0
            ]);
    } 

    static RotateXYZ(xa, ya, za)
    {
        const a = xa;
        const b = ya;
        const v = za;

        const sinA = Math.sin(a);
        const sinB = Math.sin(b);
        const sinV = Math.sin(v);
        const cosA = Math.cos(a);
        const cosB = Math.cos(b);
        const cosV = Math.cos(v);

        return new Matrix4(
            [
                cosB * cosV, sinA * sinB * cosV - cosA * sinV, cosA * sinB * cosV + sinA * sinV, 0,
                cosB * sinV, sinA * sinB * sinV + cosA * cosV, cosA * sinB * sinV - sinA * cosV, 0,
                -sinB,       sinA * cosB,                      cosA * cosB,                      0,
                0,           0,                                0,                                1.0
            ]);
    } 

    static Perspective(angleOfView = 90, imageAspectRatio = 1, near = 0.1, far = 100)
    {
        const scale = Math.tan(angleOfView * 0.5 * Math.PI / 180) * near; 
        const r = imageAspectRatio * scale;
        const l = -r; 
        const t = scale;
        const b = -t;  

        return new Matrix4(
            [
                2 * near / (r - l), 0,                  0,                              0,
                0,                  2 * near / (t - b), 0,                              0,
                (r + l) / (r - l),  (t + b) / (t - b),  -(far + near) / (far - near),   -1,
                0,                  0,                  -2 * far * near / (far - near), 0
            ]);        
    }

    static ZoomXY(scale)
    {
        return new Matrix4(
            [
                scale, 0,     0,   0,
                0,     scale, 0,   0,
                0,     0,     1.0, 0,
                0,     0,     0,   1.0
            ]);
    }     

    static Zoom(scale)
    {
        return new Matrix4(
            [
                scale, 0, 0, 0,
                0, scale, 0, 0,
                0, 0, scale, 0,
                0, 0, 0, 1.0
            ]);
    }    
}