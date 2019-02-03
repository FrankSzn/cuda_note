

// We assume col_indices and values are of length M * K
struct SparseMatrixELL {
	float* values;
	int* col_indices;
	int M;
	int N;
	int K;
};

// ELL builds on CSR with two modifications:
// 1. Padding; 2. Transposition
// Transposition: Store the sparsified matrix in a column-major format (i.e. all
// elements in the same column are in contiguous memory locations
// This is the default for FORTRAN, but not for C
__global__ void SpMV_ELL_kernle(const SparseMatricELL A, const float* x, float* y)
{
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < A.M){
		float dotProduct = 0;
		// All threads iterate the same number of times
		// Global memory access for row indices is no longer required
		for(int element = 0; element < A.K; ++element)
			// Gloat memory access depends on row, which has consecutive
			// values for consecutive threadIdx.x
			const int elementIndx = row　+ element * A.M;	
			dotProduct += A.values[elementIndex] * x[A.col_indices[elementIndex]];
		}
		y[row] = dotProduct;
	}
}
// This kernel will perform very well for matrices with similar-dense rows
// This approach is not equally well suited to all possible inputs
// Consider a 1000 X 1000 matrix with sparsity level 0.01:
// 1. There are 1000 * 100 * 0.01 = 10,000 multiply / adds to do
// 2. If the densest row has 200 nonzero values, then the kernel will perform
// 1000 * 200 = 200,000 multiply adds
// 3.By using an ELL representation, we have increased the amount of computation
// AND memory access by 20x
// 4. This is really bad worst-case performance

 
