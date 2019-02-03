// row_indices is of length M+1
// col_indices and values are of length row_induces[M]
// You are allowed to pass structs (and classes) by value to a CUDA kernel!
struct SparseMatrixCSR {
	float* values;
	int* col_indices;
	int* row_indices;
	int M;
	int N;
};


void SpMV_CSR(const SparseMatrixCSR A, const float* x, float* y){
	for(int row = 0; row < A.M; ++row){
		float dotProduct = 0;
		const int row_start = A.row_indices[row];
		const int row_end = A.row_indices[row+1];
		for(int element = row_start; element < row_end; ++element){
			dotProduct += A.values[element] * x[A.col_indices[element]];
		}
		y[row] = dotProduct;
	}
}

// As in dense matrix - vector multiplication, SpMV is data parallel
// We can compute the dot product of each row of A with x in parallel
__global__ void SpMV_CSR_kernel(const SparseMatrixCSR A, const float* x, float* y)
{
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(row < A.M){
		float dotProduct = 0;
		const int row_start = A.row_indices[row];
		const int row_end = A.row_indices[row+1];
		for(int element = row_start; element < row_end; ++element){
			dotProduct += A.values[element] * x[A.col_indices[element]];
		}
		y[row] = dotProduct;
	}
}
// Shortcoming of the Parallel SpMVCSR kernel
// 1. Non-coalesced memory access
// dotProduct += A.values[element] * x[A.col_indices[element]];
// Neighboring threads process neighboring rows, resulting in strided access
// 2. Control flow divergence
// Each flow involves a variable amount of computation, which depends on 
// the input
// 3. Additional global memory reads
// const int row_start = A.row_indices[row];
// const int row_end = A.row_indices[row+1];
// This will decrease our computation-to-global-memory-access(CGMA) ratio, which// can have significant impact for smaller and / or highly sparse matrices
