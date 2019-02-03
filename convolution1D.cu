#define TILE_SIZE 8

__global__ void convolution_1D_basic_kernel(float* N, float* M, float* P,
					int Mask_Width, int Width)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float Pvalue = 0;
	int N_start_point = i - (Mask_Width/2);

	for(int j = 0; j < Mask_Width; j++) {
		if(N_start_point + j >= 0 && N_start_point + j < Width)
		{
			Pvalue += N[N_start_point + j] * M[j];
		}
	}
	P[i] = Pvalue;
}

__global__ void convolution_1D_titled_caching_kernel(float* N, float* P,
						int Mask_Width, int Width)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float N_ds[TILE_SIZE];

	N_ds[threadIdx.x] = N[i];

	__synthreads();

	int this_tile_start_point = blockIdx.x * blockDim.x;
	int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
	int N_start_point = i - (Mask_Width / 2);
	float Pvalue = 0;
	for(int j = 0; j < Mask_width; j++){
		int N_index = N_start_point + j;
		if(N_index >= 0 && N_index < Width){
			if(N_index >= this_tile_start_point && N_index < next_tile_start_point)				{
				Pvalue += N_ds[threadIdx.x + j - (Mask_Width/2)] * M[j];
			}
			else{
				Pvalue += N[N_index] * M[j];
			}
		}
	}
	P[i] = Pvalue;
}
