

__global__ void histo_kernel_v1(unsigned char* buffer, long size, unsigned int* histo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// stride is total number of threads
	int stride = blockDim.x * gridDim.x;

	// All threads handle blockDim.x * gridDim.x
	// consectutive elements
	while(i < size){
		atomicAdd(&(histo[buffer[i]]), 1);
		i += stride;
	}

}

__global__ void histo_kernel_v2(unsigned char* buffer, long size, unsigned int* histo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// stride is total number of threads
	int stride = blockDim.x * gridDim.x;

	// All threads handle blockDim.x * gridDim.x
	// consectutive elements
	while(i < size){
		int alphabet_position = buffer[i] - "a";
		if(alphabet_position >= 0 && alpha_position < 26)	
			atomicAdd(&(histo[alphabet_postion/4]), 1);
		i += stride;
	}

}

__global__ void histo_kernel(unsigned char* buffer, long size, unsigned int* histo)
{	// Create private copies of the histo[] array for each thread block
	__shared__ unsigned int histo_private[7];
	if(threadIdx.x < 7)
		// Initialize the bin counters in the private copies of histo[]
		histo_private[threadIdx.x] = 0;
	
	__syncthreads();
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// stride is total number of threads
	int stride = blockDim.x * gridDim.x;
	while(i < size){
		atomicAdd(&(private_histo[buffer[i]/4]), 1);
		i += stride;
	}
	// wait for all other threads in the block to finish
	__syncthreads();

	if(threadIdx.x < 7){
		atomicAdd(&(histo[threadIdx.x]), private_histo[threadIdx.x]);
	}
}
