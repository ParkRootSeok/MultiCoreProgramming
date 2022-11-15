#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#pragma warning(disable:4996)

#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

char* get_source_code(const char* file_name, size_t* len) {

	char* source_code;
	char buf[2] = "\0";
	int cnt = 0;

	size_t length;
	FILE* file = fopen(file_name, "r");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);
	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);

	for (int i = 0; i < length; i++) {
		buf[0] = source_code[i];
		if (buf[0] == '\n') cnt++;
	}

	source_code[length - cnt] = '\0';
	fclose(file);
	*len = length - cnt;

	return source_code;

}

int i;

int main(int argc, char* argv[]) {

	cl_int err;

	// Platform ID
	cl_platform_id platform;
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	// Device ID
	cl_device_id device;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	/* 2. Create a context */
	cl_context context;
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	/* 3. Create a command queue */
	cl_command_queue queue;
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);

	/* 4. Create the compute program from the source buffer */
	cl_program program;
	size_t kernel_source_size;

	const char* kernel_source_code = get_source_code("kernel.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source_code, &kernel_source_size, &err);
	CHECK_ERROR(err);

	/* 5. Build the program executable */
	clBuildProgram(program, 1, &device, "", NULL, NULL);
	CHECK_ERROR(err);

	/* 6. Create the compute kernel */
	cl_kernel kernel;
	kernel = clCreateKernel(program, "vec_add", &err);
	CHECK_ERROR(err);


	/* 7. Create the input and output arrays in device memory for our calculation */

	unsigned int n = 10; // Length of vectors
	size_t bytes = n * sizeof(int); // Size, in bytes, of each vector

	cl_mem bufA, bufB; // Device input buffers
	cl_mem bufC; // Device output buffer

	int* A, *B;    // Host input vectors
	int* C;   // Host output vector

	// Allocate memory for each vector on host
	A = (int*)malloc(bytes);
	B = (int*)malloc(bytes);
	C = (int*)malloc(bytes);

	// Initialize vectors
	for (i = 0; i < n; i++) {
		A[i] = 1;
		B[i] = 1;
	}

	bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
	CHECK_ERROR(err);

	bufB = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
	CHECK_ERROR(err);

	bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
	CHECK_ERROR(err);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, bytes, A, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytes, B, 0, NULL, NULL);
	CHECK_ERROR(err);

	// print
	printf("==== CPU 계산값====\n");
	for (i = 0; i < n; i++)
		printf(" %d", A[i] + B[i]);
	printf("\n");


	/* 9. Set the arguments to our compute kernel */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
	CHECK_ERROR(err);

	/* 10. Execute the kernel over the entire range of the data set */
	size_t global_size, local_size;

	local_size = 16; // Number of work items in each local work group
	global_size = 16384;// Number of total work items

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	CHECK_ERROR(err);

	/* 8. Read the results from the device */
	err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, C, 0, NULL, NULL);
	CHECK_ERROR(err);

	// print
	printf("==== GPU 계산값====\n");
	for (i = 0; i < n; i++)
		printf(" %d", C[i]);
	printf("\n");

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// release OpenCL resources
	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	//release host memory
	free(A);
	free(B);
	free(C);

	return 0;
}