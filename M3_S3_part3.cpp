#include <stdio.h> 
#include <stdlib.h> 
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>

#else
#include <CL/cl.h>
#endif
#define VECTOR_SIZE 1024

const char *programSource =
    "__kernel void vecAdd(__global const float* a, __global const float* b, __global float* c) {\n"
    " int gid = get_global_id(0);\n"
    " c[gid] = a[gid] + b[gid];\n"
    "}\n";
int main()
{
    // Create the OpenCL context
    cl_context context = NULL;
    cl_device_id device = NULL;
    cl_command_queue commandQueue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem aMemObj = NULL;
    cl_mem bMemObj = NULL;
    cl_mem cMemObj = NULL;
    cl_int errNum;
    
    context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed to create OpenCL context\n");
        return 1;
    }
    // Get the device
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed to get OpenCL device\n");
        return 1;
    }
    // Create the command queue
    commandQueue = clCreateCommandQueue(context, device, 0, &errNum);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed to create OpenCL command queue\n");
        return 1;
    }
    // Create the program
    program = clCreateProgramWithSource(context, 1, (const char **)&programSource, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed to create OpenCL program\n");
        return 1;
    }
    // Build the program

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed to build OpenCL program\n");
        return 1;
    }
    // Create the kernel
    kernel = clCreateKernel(program, "vecAdd", &errNum);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed to create OpenCL kernel\n");
        return 1;
    }
    // Allocate memory for the vectors
    float *a = (float *)malloc(sizeof(float) * VECTOR_SIZE);
    float *b = (float *)malloc(sizeof(float) * VECTOR_SIZE);
    float *c = (float *)malloc(sizeof(float) * VECTOR_SIZE);
    srand(time(NULL));
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        a[i] = (float)rand() / (float)RAND_MAX;
        b[i] = (float)rand() / (float)RAND_MAX;
    }
    aMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * VECTOR_SIZE, a, &errNum);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed to create OpenCL buffer\n");
        return 1;
    }
    bMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * VECTOR_SIZE, b, &errNum);
    if (errNum != CL_SUCCESS)
    {

        printf("Failed to create OpenCL buffer\n");
        return 1;
    }
    // Set the kernel arguments
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMemObj);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMemObj);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMemObj);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed to set OpenCL kernel arguments\n");
        return 1;
    }
    // Execute the kernel
    size_t globalWorkSize[1] = {VECTOR_SIZE};
    size_t localWorkSize[1] = {1};
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed to enqueue OpenCL kernel\n");
        return 1;
    }
    // Read the output buffer
    errNum = clEnqueueReadBuffer(commandQueue, cMemObj, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, c, 0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed to read OpenCL buffer\n");
        return 1;
    }
    // Verify the result
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        if (c[i] != a[i] + b[i])
        {
            printf("Error: mismatch at index %d\n", i);

            return 1;
        }
    }
    printf("Vector addition successful\n");
    // Cleanup
    clReleaseMemObject(aMemObj);
    clReleaseMemObject(bMemObj);
    clReleaseMemObject(cMemObj);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    free(a);
    free(b);
    free(c);
    return 0;
}