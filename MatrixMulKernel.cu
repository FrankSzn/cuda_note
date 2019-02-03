
#define TILE_WIDTH 16
#define BLOCK_WIDTH 16



__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;	int by = blockIdx.y;
	int tx = threadIdx.x;	int ty = threadIdx.y;

	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;

	// Loop over the M and N tiles required to compute the P element
	for(int p = 0; p < Width/TILE_WIDTH; ++p)
	{
		// Collaborative loading of M and N tiles into shared memory
		ds_M[ty][tx] = M[Row*Width + p*TILE_WIDTH+tx];
		ds_N[ty][tx] = N[(p*TILE_WIDTH+ty)*Width + Col];
		__syncthreads();

		for(int i = 0; i < TILE_WIDTH; ++i)\
			Pvalue += ds_M[ty][i] * ds_N[i][tx];
		__syncthreads();
	}
	P[Row*Width+Col] = Pvalue;
}


// An alternative tiled multiplication kernel
__global__ void MatrixMulKernel2(float* M, float* N, floar* P, int Width)
{
	__shared__ float ds_M[BLOCK_WIDTH][BLOCK_WIDTH/2];
	__shared__ float ds_N[BLOCK_WIDTH/2][BLOCK_WIDTH];

	int bx = blockIdx.x;	int by = blockIdx.y;
	int tx = threadIdx.x;	int ty = threadIdx.y;

	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;
	float Pvalue = 0;

	// loop over the M and N tiles required to compute the P element
	for(int p = 0; p < 2 * Width / BLOCK_WIDTH; ++p){
		// Collaborative loading of M and N tiles into shared memory
		if(tx < BLOCK_WIDTH/2 ){
			ds_M[ty][tx] = M[Row*Width + p*BLOCK_WIDTH/2 + tx];
		}else{
			ds_N[tx - BLOCK_WIDTH/2][ty] = N[p*BLOCK_WIDTH/2 + (tx - BLOCK_WIDTH/2)*Width + bx * blockDim.x + ty];
		}
		
		__synthreads();

		for(int i = 0; i < BLOCK_WIDTH/2; ++i)
		{
			Pvalue += ds_M[ty][i] * ds_N[i][tx];
		}
		__synthreads();

	}
	P[Row*Width+Col] = Pvalue;
}


// no good this one
__global__ void MatrixMulKernel3(float* M, float* N, float* P, int Width)
{
	__shared__ float ds_M[BLOCK_WIDTH];
	__shared__ float ds_N[BLOCK_WIDTH];

	int bx = blockIdx.x;	int by = blockIdx.y;
	int tx = threadIdx.x;	int ty = threadIdx.y;

	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;
	float Pvalue = 0;

	// Loop over the M and N tiles required to compute the P element
	for(int p = 0; p < Width; ++p)	{
		// Collaborative loading of M and N tiles into shared memory
		if(tx == 0){
			ds_M[ty] = M[Row*Width + k];
		}else{
			ds_N[ty] = N[p * Width + bx * blockDim.x + ty];
		}
		__syncthreads();

		Pvalue += ds_M[ty] * ds_N[tx];
		__synthreads();
	}
	P[Row*Width + Col] = Pvalue;
}

// Tile with boundary checking
__global__ void MatrixMulKernel4(float* M, float* N, float* P, int Width)
{
	__shared__ float ds_M[BLOCK_WIDTH];
	__shared__ float ds_N[BLOCK_WIDTH];

	int bx = blockIdx.x;	int by = blockIdx.y;
	int tx = threadIdx.x;	int ty = threadIdx.y;

	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;
	float Pvalue = 0;

	for(int p = 0; p < (Width-1)/ TILE_WIDTH + 1; ++p){
		if(Row < Width && p * TILE_WIDTH + tx < Width ){
			ds_M[ty][tx] = M[Row * Width + p * TILE_WIDTH + tx];
		}else {
			ds_M[ty][tx] = 0.0;	
		}
		
		if(p*TILE_WIDTH + ty < Width && Col < Width){
			ds_N[ty][tx] = N[(p*TILE_WIDTH+ ty) * Width + Col];
		}else {
			ds_N[ty][tx] = 0.0;	
		}
		__syncthreads();
		if(Row < Width && Col < Width){
			for(nt i = 0; i < TILE_WIDTH; ++i){
				Pvalue += ds_M[ty][i] * ds_N[i][tx];
			}
			__synthreads();
		}
	}// end of outer for loop
	if(Row < Width && Col < Width)
		P[Row*Width + Col] = Pvalue;
}

