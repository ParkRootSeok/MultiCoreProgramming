#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include "rotation.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

char* get_source_code(const char* file_name, size_t* len) {
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char* source_code = (char*)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    };
}

void rotate(const float* input, float* output, const int width, const int height, char *degree) {
    
	const float theta = atof(degree) * M_PI / 180;
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    cl_int err;

    // Platform ID
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    // Device ID
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    // Create Context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    // Create Command Queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    CHECK_ERROR(err);

    // Create Program Object
    size_t kernel_source_size;
    char* kernel_source = get_source_code("kernel.cl", &kernel_source_size);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    // Build Program
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    build_error(program, device, err);
    CHECK_ERROR(err);

	/* 6. Create the compute kernel */
	cl_kernel kernel;

	kernel = clCreateKernel(program, "image_rotate", &err);
	CHECK_ERROR(err);

	/* 7. Create the input and output arrays in device memory for our calculation */
	cl_mem bufinput; 
	cl_mem bufoutput; 

	bufinput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height, NULL, NULL);
	CHECK_ERROR(err);

	bufoutput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height, NULL, NULL);
	CHECK_ERROR(err);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, bufinput, CL_TRUE, 0, sizeof(float) * width * height, input, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, bufoutput, CL_TRUE, 0, sizeof(float) * width * height, output, 0, NULL, NULL);
	CHECK_ERROR(err);

	/* 8. Set the arguments to our compute kernel */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufinput);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufoutput);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_int), &width);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 3, sizeof(cl_int), &height);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 4, sizeof(cl_char), degree);
	CHECK_ERROR(err);


	/* 9. Execute the kernel over the entire range of the data set */
	size_t local_size;  // Number of work items in each local work group
	size_t global_size[2] = { width, height }; // Number of total work items

	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, &global_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);

	/* 10. Read the results from the device */
	err = clEnqueueReadBuffer(queue, bufoutput, CL_TRUE, 0, sizeof(float) * width * height, output, output, 0, NULL, NULL);
	CHECK_ERROR(err);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// release OpenCL resources
	clReleaseMemObject(bufinput);
	clReleaseMemObject(bufoutput);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}