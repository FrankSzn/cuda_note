#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>


#define IDX2C(i,j, ld) (((j)*(ld))+(i))

// Matrix size
#define N (275)

//a[IDX2C(0, 1, 50)]

// Host implementation of a simple version of sgemm
static void simple_sgemm(int n, float alpha, const float* A, const float* B,
			float beta, float* C)
{
	int i;
	int j;
	int k;

	for ( i = 0; i < n; ++i)
	{
		for(j = 0; j < n; ++j)
		{
			float prod = 0;
			
			for( k = 0; k < n; ++k)
			{
				prod += A[k * n + i] * B[j * n + k];
			}
			C[j * n + i] = alpha * prod + beta * C[j * n + i];
		}
	}
}

// Main
int main(int argc, char **argv)
{
	cublasStatus_t status;
	float* h_A;
	float* h_B;
	float* h_C;
	float* h_C_ref;

	float* d_A = 0;
	float* d_B = 0;
	float* d_C = 0;

	float alpha = 1.0f;
	float beta = 0.0f;
	int n2 = N * N;
	int i;
	
	float error_norm;
	float ref_norm;
	float diff;
	cublasHandle_t handle;

	status = cublasCreate(&handle);

	// Allocate host memory for the matrices
	h_A = (float *)malloc(n2 * sizeof(h_A[0]));
	h_B = (float *)malloc(n2 * sizeof(h_B[0]));
	h_C = (float *)malloc(n2 * sizeof(h_C[0]));

	// Fill the matrix with test data
	for ( i = 0 ; i < n2; i++)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
		h_C[i] = rand() / (float)RAND_MAX;
	}
	
	// Allocate device memory for the matrices
	cudaMalloc((void**)&d_A, n2 * sizeof(d_A[0]));
	cudaMalloc((void**)&d_B, n2 * sizeof(d_B[0]));
	cudaMalloc((void**)&d_C, n2 * sizeof(d_C[0]));

	// Initialize the device matrices with the host matrices
	status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
	status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
	status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);

	// Performs operation using plain C code
	simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
	h_C_ref = h_C;

	// Performs operation using cublas
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N
				, &alpha, d_A, N, d_B, N, &beta, d_C, N);

	// Allocate host memory for reading back the result from device memory
	h_C = (float*)malloc(n2 * sizeof(h_C[0]));

	// Read the result back
	status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

	// Check result against reference
	error_norm = 0;
	ref_norm = 0;

	for(i = 0; i < n2; ++i)
	{
		diff = h_C_ref[i] - h_C[i];	
		error_norm += diff * diff;
		ref_norm += h_C_ref[i] * h_C_ref[i];
	}

	error_norm = (float)sqrt((double)error_norm);
	ref_norm = (float)sqrt((double)ref_norm);

	// Memory clean up
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_ref);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	printf("%f", error_norm);
	// shutdown
	status = cublasDestroy(handle);
}
