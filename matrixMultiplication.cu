

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
	// Calculate the row index of the P element and M
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	// Calculate the column index of P and N
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if((Row < Width) && (Col < Width))	{
		float Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for(int k = 0; k < Width; ++k) {
			Pvalue += M[Row*Width + k] * N[k * Width + Col];
		}
		P[Row*Width + Col] = Pvalue;
	}
}

__global__ void TiledMatrixMulKernel()
