__kernel void mat_mul_seq (__global int **A, __global int **B, __global int **C,
							__global int row_a, __global int col_a, __global int col_b) {
	
	int i = get_global_id(0);	// row index
	int j = get_global_id(1);	// col index

	int k, sum = 0;
	for (k = 0;k < col_b;k++) {
		sum += A[i * col_b + k] * B[k * col_b + j];
	}
	 
	C[i * col_b + j] = sum;
	
}