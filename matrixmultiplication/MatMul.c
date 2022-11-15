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

	cl_program program;	// program
	cl_kernel kernel;	// kernel
	cl_int err;

	/* 1. read device */
	cl_uint num_platforms;
	cl_platform_id* platforms;
	cl_uint num_devices;
	cl_device_id* device;
	char str[1024];
	cl_device_type device_type;
	size_t max_work_group_size;
	cl_uint max_clock_frequency;
	cl_ulong global_mem_size;
	cl_ulong local_mem_size;
	cl_ulong max_mem_alloc_size;
	cl_ulong max_compute_units;
	cl_command_queue_properties queue_properties;
	cl_uint p, d;

	err = clGetPlatformIDs(0, NULL, &num_platforms);   // Bind to platform
	CHECK_ERROR(err);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);

	err = clGetPlatformIDs(num_platforms, platforms, NULL); // Get ID for the device
	CHECK_ERROR(err);

	printf("Number of platforms: %u\n\n", num_platforms);
	for (p = 0; p < num_platforms; p++)
	{
		printf("platform: %u\n", p);

		err = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, str, NULL);
		CHECK_ERROR(err);
		printf("- CL_PLATFORM_NAME\t:%s\n", str);

		err = clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 1024, str, NULL);
		CHECK_ERROR(err);
		printf("- CL_PLATFORM_VENDOR\t:%s\n\n", str);


		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		CHECK_ERROR(err);
		printf("Number of devices:\t%u\n\n", num_devices);

		device = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, device, NULL);
		CHECK_ERROR(err);

		for (d = 0; d < num_devices; d++)
		{
			printf("device: %u\n", d);

			err = clGetDeviceInfo(device[d], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_TYPE\t:");
			if (device_type & CL_DEVICE_TYPE_CPU) printf(" CL_DEVICE_TYPE_CPU");
			if (device_type & CL_DEVICE_TYPE_GPU) printf(" CL_DEVICE_TYPE_GPU");
			if (device_type & CL_DEVICE_TYPE_ACCELERATOR) printf(" CL_DEVICE_TYPE_ACCELERATOR");
			if (device_type & CL_DEVICE_TYPE_DEFAULT) printf(" CL_DEVICE_TYPE_DEFAULT");
			if (device_type & CL_DEVICE_TYPE_CUSTOM) printf(" CL_DEVICE_TYPE_CUSTOM");
			printf("\n");

			err = clGetDeviceInfo(device[d], CL_DEVICE_NAME, 1024, str, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_NAME\t: %s\n", str);

			err = clGetDeviceInfo(device[d], CL_DEVICE_VENDOR, 1024, str, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_VENDOR\t: %s\n", str);

			err = clGetDeviceInfo(device[d], CL_DEVICE_VERSION, 1024, str, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_VERSION\t: %s\n", str);

			err = clGetDeviceInfo(device[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_ulong), &max_clock_frequency, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_MAX_CLOCK_FREQUENCY : %luMHz\n", max_clock_frequency);

			err = clGetDeviceInfo(device[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_ulong), &max_compute_units, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_MAX_COMPUTE_UNITS : %lu\n", max_compute_units);

			err = clGetDeviceInfo(device[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_MAX_WORK_GROUP_SIZE : %lu\n", max_work_group_size);

			err = clGetDeviceInfo(device[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_GLOBAL_MEM_SIZE : %lu\n", global_mem_size);

			err = clGetDeviceInfo(device[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_LOCAL_MEM_SIZE : %lu\n", local_mem_size);

			err = clGetDeviceInfo(device[d], CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_properties, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_QUEUE_PROPERTIES :");
			if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) printf(" CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
			if (queue_properties & CL_QUEUE_PROFILING_ENABLE) printf(" CL_QUEUE_PROFILING_ENABLE");
			printf("\n");
		}

	}

	/* 2. Create a context */
	cl_context context;
	context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
	CHECK_ERROR(err);

	/* 3. Create a command queue */
	cl_command_queue queue;
	queue = clCreateCommandQueueWithProperties(context, device[0], 0, &err);
	CHECK_ERROR(err);

	/* 4. Create the compute program from the source buffer */
	size_t kernel_source_size;

	const char* kernel_source_code = get_source_code("kernel.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source_code, &kernel_source_size, &err);
	CHECK_ERROR(err);

	/* 5. Build the program executable */
	clBuildProgram(program, 1, device, "", NULL, NULL);
	CHECK_ERROR(err);

	/* 6. Create the compute kernel */
	kernel = clCreateKernel(program, "mat_mul_seq", &err);
	CHECK_ERROR(err);

	/* 7. Create the input and output arrays in device memory for our calculation */
	int rowA = 1000, colA = 1000, colB = 1000;

	cl_mem bufA, bufB; // Device input buffers
	cl_mem bufC; // Device output buffer

	int *A,*B;    // host input matrix
	int *C;		// host output Matrix

	// Allocate memory for each Matrix on host
	A = (int*)malloc(sizeof(int) * rowA * colA);
	B = (int*)malloc(sizeof(int) * colA * colB);
	C = (int*)malloc(sizeof(int) * rowA * colB);


	// Initialize Matrix
	for (int i = 0; i < rowA; i++) {
		for (int j = 0; j < colB; j++) {
			A[i * rowA + j] = 1;
			B[i * rowA + j] = 1;
		}
	}

	bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, rowA * colB * sizeof(int), NULL, NULL);
	CHECK_ERROR(err);

	bufB = clCreateBuffer(context, CL_MEM_READ_WRITE, rowA * colB * sizeof(int), NULL, NULL);
	CHECK_ERROR(err);

	bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, rowA * colB * sizeof(int), NULL, NULL);
	CHECK_ERROR(err);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, rowA * colB * sizeof(int), A, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, rowA * colB * sizeof(int), B, 0, NULL, NULL);
	CHECK_ERROR(err);


	// Matrix Mul
	for (int i = 0; i < rowA; i++) {

		for (int j = 0; j < colB; j++) {
			C[i * colB + j] = 0;
			for (int k = 0; k < colB; k++) {
				C[i * colB + j] += A[i * colB + k] * B[k * colB + i];
			}
		}
	} printf("\n");

	/* 8. Set the arguments to our compute kernel */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 3, sizeof(cl_int), &rowA);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 4, sizeof(cl_int), &colA);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 5, sizeof(cl_int), &colB);
	CHECK_ERROR(err);


	/* 9. Execute the kernel over the entire range of the data set */
	
	size_t local_size;  // Number of work items in each local work group
	size_t global_size[2] = {rowA, colA}; // Number of total work items

	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, &global_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);

	/* 10. Read the results from the device */
	err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, rowA * colB * sizeof(int), C, 0, NULL, NULL);
	CHECK_ERROR(err);

	for (int i = 0; i < rowA; i++) {
		for (int j = 0;j < colB;j++) {
			printf(" %d", *(C + (i * colB + j)));
		} printf("\n");
	} printf("\n");


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