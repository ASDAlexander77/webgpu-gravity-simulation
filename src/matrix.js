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
}

class Matrix4
{
    static Create()
    {
        return new Matrix(
            4, 4,
            [
                1.0, 0, 0, 0,
                0, 1.0, 0, 0,
                0, 0, 1.0, 0,
                0, 0, 0, 1.0
            ]);
    }
}