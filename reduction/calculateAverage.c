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


int main() {

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
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source_code, &kernel_source_size, &err);
	CHECK_ERROR(err);

	/* 5. Build the program executable */
	clBuildProgram(program, 1, &device, "", NULL, NULL);
	CHECK_ERROR(err);

	/* 6. Create the compute kernel */
	cl_kernel kernel;
	kernel = clCreateKernel(program, "reduction", &err);
	CHECK_ERROR(err);

	/* 7. Create the input and output arrays in device memory for our calculation */

	cl_mem numbersBuffer, sumBuffer;

	cl_int capacity = 512;
	int* numbers, * sum;

	size_t numbersSize = capacity * sizeof(int);
	size_t sumSize = sizeof(int);

	// Allocate memory for each Matrix on host
	numbers = (int*)malloc(sizeof(int) * capacity);
	sum = (int*)malloc(sizeof(int));

	// Initialize Matrix
	for (int i = 0; i < capacity; i++) {
		numbers[i] = 1;
	}

	numbersBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, numbersSize, NULL, NULL);
	CHECK_ERROR(err);

	sumBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sumSize, NULL, NULL);
	CHECK_ERROR(err);


	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, numbersBuffer, CL_TRUE, 0, numbersSize, numbersBuffer, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, sumBuffer, CL_TRUE, 0, sumSize, sumBuffer, 0, NULL, NULL);
	CHECK_ERROR(err);


	// Calculate Average
	int result = 0;
	for (int i = 0; i < capacity; i++) {
		result += numbers[i];
	} result /= capacity;
	printf("Result by not parallel : %d \n", result);

	/* 8. Set the arguments to our compute kernel */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &numbersBuffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &sumBuffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_int), &capacity);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 3, sizeof(int) * 256, NULL);
	CHECK_ERROR(err);

	/* 9. Execute the kernel over the entire range of the data set */

	size_t local_size = 256;  // Number of work items in each local work group
	size_t global_size = capacity; // Number of total work items

	err = clEnqueueNDRangeKernel(queue, kernel, 1, &local_size, &global_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);

	/* 10. Read the results from the device */
	err = clEnqueueReadBuffer(queue, sumBuffer, CL_TRUE, 0, sumSize, sum, 0, NULL, NULL);
	CHECK_ERROR(err);
	
	//printf("Result by parallel : %d\n", sum);


	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// release OpenCL resources
	clReleaseMemObject(numbersBuffer);
	clReleaseMemObject(sumBuffer);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	//release host memory
	free(numbers);
	free(sum);

	return 0;
}