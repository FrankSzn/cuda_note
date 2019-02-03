

struct SparseMatrixCSR
{
	/* data */
	float* values;
	int* col_indices;	
	int* row_indices; // row_indices is of length M+1
	int M; // number of rows in the matrix
	int N; // number of columns in the matrix
};


