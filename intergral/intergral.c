#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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

double f(double x) {

	return ((3 * x * x) + (2 * x) + 1);

}

double integral(int N) {

	double dx = (1.0) / (double)N;
	double sum = 0;

	for (int i = 0; i < N; i++) {
		sum += f((double)(i)*dx) * dx;
	}

	return sum;

}

int main() {

	clock_t start, end;

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

	const char* kernel_source_code = get_source_code("int_kernel.cl", &kernel_source_size);
	// printf("%s", kernel_source_code);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source_code, &kernel_source_size, &err);
	CHECK_ERROR(err);

	/* 5. Build the program executable */
	clBuildProgram(program, 1, &device, "", NULL, NULL);
	CHECK_ERROR(err);

	/* 6. Create the compute kernel */
	cl_kernel kernel;
	kernel = clCreateKernel(program, "integral", &err);
	CHECK_ERROR(err);

	int N = 1000;
	float result;

	/* host Calculate Average */
	start = clock();
	result = integral(N);
	end = clock();
	printf("Not parallel running time : %.f sec \n", (double)(end - start) / CLK_TCK);
	printf("Not parallel Result : %.4f\n\n", result);

	/* 7. Create the input and output arrays in device memory for our calculation */

	float* sum = (float*)malloc(N * sizeof(float));

	for (int i = 0; i < N; i++) {
		sum[i] = 0;
	}


	cl_mem sumBuffer;
	size_t sumSize = N * sizeof(float);

	sumBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sumSize, NULL, NULL);
	CHECK_ERROR(err);

	/* 8. Set the arguments to our compute kernel */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sumBuffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_int), &N);
	CHECK_ERROR(err);

	/* 9. Execute the kernel over the entire range of the data set */
	size_t local_size = 1;  // Number of work items in each local work group
	size_t global_size = N; // Number of total work items

	start = clock();
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);
	clFinish(queue);
	end = clock();
	printf("parallel running time : %.f sec \n", (double)(end - start) / CLK_TCK);

	/* 10. Read the results from the device */
	err = clEnqueueReadBuffer(queue, sumBuffer, CL_TRUE, 0, sumSize, sum, 0, NULL, NULL);
	CHECK_ERROR(err);
	
	result = 0;
	for (int i = 0; i < N; i++) {
		result += sum[i];
	} printf("Parallel Result : %.4f \n", result);
	

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// release OpenCL resources
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	//release host memory

	return 0;
}