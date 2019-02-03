
// We assume row_indices, col_indices, and values are of length count
struct SparseMatrixCOO {
	float* values;
	int* col_indices;
	int* row_indices;
	int M;
	int N;
	int count;
};

// Compared to the sequential SpMV/CSR, the sequential SpMN/COO doesn't waste
// time with fully-zero rows
void SpMV_COO(const SparseMatrixCOO A, const float* x, float* y){
	for(int element = 0; element < A.count; ++element){
		const int column = A.col_indices[element];
		const int row = A.row_indices[element];

		y[row] += A.values[element] * x[column];
	}
}

__global__ void SpMV_COO_kernel_v1(const SparseMatrixCOO A, const float* x, float* y)
{
	for(int element = threadIdx.x + blockIdx.x * blockDim.x;
		element < A.count;
		element += blockDim.x * gridDim.x){
		
		const int column = A.col_indices[element];
		const int row = A.row_indices[element];
		// Output interference		
		y[row] += A.values[element] * x[column];
	}
}

// Swithcing to an atomic addition will make the output of this kernel coorect
// It will also serializat a potential large number of writes
// We could solve this using techniques from the histogram pattern(i.e.
// privitization)
// We'll note that this representation is better suited to sequential hardware
// and take a different approach.
__global__ void SpMV_COO_kernel_v2(const SparseMatrixCOO A, const float* x, float* y)
{
	for(int element = threadIdx.x + blockIdx.x * blockDim.x;
		element < A.count;
		element += blockDim.x * gridDim.x){
		
		const int column = A.col_indices[element];
		const int row = A.row_indices[element];
		
		atomicAdd(&y[row], A.values[element] * x[column]);	
	}
}
