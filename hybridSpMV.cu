

struct SparseMatrixELL{
	float* values;
	int* col_indices;
	int M;
	int N;
	int K;	
};

struct SparseMatrixCOO{
	float* values;
	int* col_indices;
	int* row_indices;
	int M;
	int N;
	int count;
};



__global__ void SpMV_ELL_kernel(const SparseMatrixELL A, const float* x, float* y)
{
i	const int row = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < A.M){
		float dotProduct = 0;

		for(int element = 0; element < A.K; ++element){
			const int elementIndex = row + element * A.M;
			dotProuct += A.values[elementIndex] * x[A.col_indices[elementIndx]];
		}
		
		y[row] = dotProduct;
	}
}

void SpMV_COO(const SparseMatrixCOO A, const float* x, float* y){
	for(int element = 0; element < A.count; ++element){
		const int column = A.col_indices[element];
		const int row = A.row_indices[element];

		y[row] += A.values[element] * x[column];
	}
}

void hybridSpMV(const float* A, const int M, const int N, const float* x, float* y) {
	float* d_y;
	float* d_x;
	float* y_ELL;
	SparseMatrixELL d_A_ELL;
	SparseMatrixCOO A_COO;

	// build sparse matrix representation, allocate / initialize host and device
	// memory
	malloc(d_x, 0, sizeof(d_x));
	malloc(d_y, 0, sizeof(d_y));
	cudaMalloc(&d_x, sizeof());
	

	
	// launch ELL kernel
	SpMV_ELL_kernel<<<(A.M + 127)/128, 128>>>(d_A_ELL, d_x, d_y);
	// copy device result back to host
	cudaMemcpy(y_ELL, d_y, A.N * sizeof(float), cudaMemcpyDeviceToHost);
	// perform host computation
	SpMV_COO(A_COO, x, y_ELL);
}


