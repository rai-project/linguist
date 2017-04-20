package linguist

import (
	"github.com/GeertJohan/go.rice/embedded"
	"time"
)

func init() {

	// define files
	file2 := &embedded.EmbeddedFile{
		Filename:    `cpp_sample.cpp`,
		FileModTime: time.Unix(1492675663, 0),
		Content:     string("#include <wb.h>\n\n//@@ The purpose of this code is to become familiar with the submission\n//@@ process. Do not worry if you do not understand all the details of\n//@@ the code.\n\nint main(int argc, char **argv) {\n  int deviceCount;\n\n  wbArg_read(argc, argv);\n\n  cudaGetDeviceCount(&deviceCount);\n\n  wbTime_start(GPU, \"Getting GPU Data.\"); //@@ start a timer\n\n  for (int dev = 0; dev < deviceCount; dev++) {\n    cudaDeviceProp deviceProp;\n\n    cudaGetDeviceProperties(&deviceProp, dev);\n\n    if (dev == 0) {\n      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {\n        wbLog(TRACE, \"No CUDA GPU has been detected\");\n        return -1;\n      } else if (deviceCount == 1) {\n        //@@ WbLog is a provided logging API (similar to Log4J).\n        //@@ The logging function wbLog takes a level which is either\n        //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or TRACE and a\n        //@@ message to be printed.\n        wbLog(TRACE, \"There is 1 device supporting CUDA\");\n      } else {\n        wbLog(TRACE, \"There are \", deviceCount,\n              \" devices supporting CUDA\");\n      }\n    }\n\n    wbLog(TRACE, \"Device \", dev, \" name: \", deviceProp.name);\n    wbLog(TRACE, \" Computational Capabilities: \", deviceProp.major, \".\",\n          deviceProp.minor);\n    wbLog(TRACE, \" Maximum global memory size: \",\n          deviceProp.totalGlobalMem);\n    wbLog(TRACE, \" Maximum constant memory size: \",\n          deviceProp.totalConstMem);\n    wbLog(TRACE, \" Maximum shared memory size per block: \",\n          deviceProp.sharedMemPerBlock);\n    wbLog(TRACE, \" Maximum block dimensions: \",\n          deviceProp.maxThreadsDim[0], \" x \", deviceProp.maxThreadsDim[1],\n          \" x \", deviceProp.maxThreadsDim[2]);\n    wbLog(TRACE, \" Maximum grid dimensions: \", deviceProp.maxGridSize[0],\n          \" x \", deviceProp.maxGridSize[1], \" x \",\n          deviceProp.maxGridSize[2]);\n    wbLog(TRACE, \" Warp size: \", deviceProp.warpSize);\n  }\n\n  wbTime_stop(GPU, \"Getting GPU Data.\"); //@@ stop the timer\n\n  return 0;\n}"),
	}
	file3 := &embedded.EmbeddedFile{
		Filename:    `cuda_sample.cu`,
		FileModTime: time.Unix(1492675663, 0),
		Content:     string("#include <wb.h>\n\n__global__ void vecAdd(float *in1, float *in2, float *out, int len) {\n  //@@ Insert code to implement vector addition here\n  int index = threadIdx.x + blockIdx.x * blockDim.x;\n  if (index < len) {\n    out[index] = in1[index] + in2[index];\n  }\n}\n\nint main(int argc, char **argv) {\n  wbArg_t args;\n  int inputLength;\n  float *hostInput1;\n  float *hostInput2;\n  float *hostOutput;\n  float *deviceInput1;\n  float *deviceInput2;\n  float *deviceOutput;\n\n  args = wbArg_read(argc, argv);\n\n  wbTime_start(Generic, \"Importing data and creating memory on host\");\n  hostInput1 =\n      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);\n  hostInput2 =\n      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);\n  hostOutput = (float *)malloc(inputLength * sizeof(float));\n  wbTime_stop(Generic, \"Importing data and creating memory on host\");\n\n  wbLog(TRACE, \"The input length is \", inputLength);\n\n  wbTime_start(GPU, \"Allocating GPU memory.\");\n  //@@ Allocate GPU memory here\n  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(float));\n  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(float));\n  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(float));\n  wbTime_stop(GPU, \"Allocating GPU memory.\");\n\n  wbTime_start(GPU, \"Copying input memory to the GPU.\");\n  //@@ Copy memory to the GPU here\n  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float),\n             cudaMemcpyHostToDevice);\n  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float),\n             cudaMemcpyHostToDevice);\n  wbTime_stop(GPU, \"Copying input memory to the GPU.\");\n\n  //@@ Initialize the grid and block dimensions here\n  dim3 blockDim(32);\n  dim3 gridDim(ceil(((float)inputLength) / ((float)blockDim.x)));\n\n  wbLog(TRACE, \"Block dimension is \", blockDim.x);\n  wbLog(TRACE, \"Grid dimension is \", gridDim.x);\n\n  wbTime_start(Compute, \"Performing CUDA computation\");\n  //@@ Launch the GPU Kernel here\n  vecAdd<<<gridDim, blockDim>>>(deviceInput1, deviceInput2, deviceOutput,\n                                inputLength);\n  cudaDeviceSynchronize();\n  wbTime_stop(Compute, \"Performing CUDA computation\");\n\n  wbTime_start(Copy, \"Copying output memory to the CPU\");\n  //@@ Copy the GPU memory back to the CPU here\n  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float),\n             cudaMemcpyDeviceToHost);\n  wbTime_stop(Copy, \"Copying output memory to the CPU\");\n\n  wbTime_start(GPU, \"Freeing GPU Memory\");\n  //@@ Free the GPU memory here\n  cudaFree(deviceInput1);\n  cudaFree(deviceInput2);\n  cudaFree(deviceOutput);\n  wbTime_stop(GPU, \"Freeing GPU Memory\");\n\n  wbSolution(args, hostOutput, inputLength);\n\n  free(hostInput1);\n  free(hostInput2);\n  free(hostOutput);\n\n  return 0;\n}"),
	}
	file4 := &embedded.EmbeddedFile{
		Filename:    `openacc_sample.cpp`,
		FileModTime: time.Unix(1492675663, 0),
		Content:     string("#include <wb.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n \nint main( int argc, char* argv[] ) {\n    wbArg_t args;\n    float * __restrict__ input1;\n    float * __restrict__ input2;\n    float * __restrict__ output;\n    int inputLength;\n\n    args = wbArg_read(argc, argv);\n \n    wbTime_start(Generic, \"Importing data and creating memory on host\");\n    input1 =\n        (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);\n    input2 =\n        (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);\n    output = (float *)malloc(inputLength * sizeof(float));\n    wbTime_stop(Generic, \"Importing data and creating memory on host\");\n\n    //@@ Insert vector addition code here.\n    wbTime_start(GPU, \"Copy to GPU, compute, and copy back to host.\");\n    int i;\n    // sum component wise and save result into vector c\n    #pragma acc kernels copyin(input1[0:inputLength],input2[0:inputLength]), copyout(output[0:inputLength])\n    for(i = 0; i < inputLength; ++i) {\n        output[i] = input1[i] + input2[i];\n    }\n    wbTime_stop(GPU, \"Copy to GPU, compute, and copy back to host.\");\n\n    wbSolution(args, output, inputLength);\n \n    // Release memory\n    free(input1);\n    free(input2);\n    free(output);\n \n    return 0;\n}"),
	}
	file5 := &embedded.EmbeddedFile{
		Filename:    `opencl_sample.cpp`,
		FileModTime: time.Unix(1492675663, 0),
		Content:     string("#include <wb.h> //@@ wb include opencl.h for you\n#include <math.h>\n//@@ OpenCL Kernel\nconst char* vaddsrc =\"__kernel void vadd(__global const float *a,__global const float *b,__global float *result){int id = get_global_id(0);result[id] = a[id] + b[id];}\";\n\nint main(int argc, char **argv)\n{\n\n    unsigned int VECTOR_SIZE = 1024;\n    int size = VECTOR_SIZE* sizeof(float);\n    wbArg_t args;\n    int inputLength = VECTOR_SIZE;\n    float *hostInput1;\n    float *hostInput2;\n    float *hostOutput;\n    cl_mem deviceInput1;\n    cl_mem deviceInput2;\n    cl_mem deviceOutput;\n\n  args = wbArg_read(argc, argv);\n  wbTime_start(Generic, \"Importing data and creating memory on host\");\n  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);\n  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);\n  hostOutput = (float *)malloc(inputLength * sizeof(float));\n   wbTime_stop(Generic, \"Importing data and creating memory on host\");\n  wbLog(TRACE, \"The input length is \", inputLength);\n\n\n  //@@ OpenCL Context Setup Code (simple)\n        size_t parmsz;\n        cl_int clerr;\n        cl_context clctx;\n        cl_command_queue clcmdq;\n        cl_program clpgm;\n        cl_kernel clkern;\n\n// query the number of platforms\n   cl_uint numPlatforms;\n   clerr = clGetPlatformIDs(0, NULL, &numPlatforms);\n   cl_platform_id platforms[numPlatforms];\n   clerr = clGetPlatformIDs(numPlatforms, platforms, NULL);\n   cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (unsigned long)platforms[0], 0};\n   clctx = clCreateContextFromType(properties,CL_DEVICE_TYPE_ALL, NULL, NULL, &clerr);\n    clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0,NULL, & parmsz);\n   cl_device_id* cldevs = (cl_device_id *) malloc( parmsz);\n   clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz,cldevs, NULL);\n   clcmdq = clCreateCommandQueue(clctx,cldevs[0], 0, &clerr);\n   clpgm = clCreateProgramWithSource(clctx, 1, &vaddsrc,NULL, &clerr);\n   char clcompileflags[4096];\n   sprintf(clcompileflags, \"-cl-mad-enable\");\n   clerr = clBuildProgram(clpgm, 0, NULL, clcompileflags,NULL, NULL);\n   clkern = clCreateKernel(clpgm, \"vadd\", &clerr);\n  //@@ OpenCL Context Setup Code (simple)\n\n\n  wbTime_start(GPU, \"Allocating GPU memory.Copying input memory to the GPU.\");\n  //@@ Allocate GPU memory here Copy memory to the GPU here\n  deviceInput1 = clCreateBuffer(clctx, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR,inputLength *sizeof(float), hostInput1, NULL);\n  deviceInput2 = clCreateBuffer(clctx, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR, inputLength *sizeof(float), hostInput2, NULL);\n  deviceOutput = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY,inputLength *sizeof(float), NULL, NULL);\n  wbTime_stop(GPU, \"Allocating GPU memory.Copying input memory to the GPU.\");\n\n\n\n  //@@ Initialize the grid and block dimensions here\nsize_t globalSize, localSize;\nlocalSize = 64;\nglobalSize = ceil(inputLength/(float)localSize)*localSize;\n\n\n\n\n  wbTime_start(Compute, \"Performing CUDA computation\");\n  //@@ Launch the GPU Kernel here\nclerr= clSetKernelArg(clkern, 0, sizeof(cl_mem),(void *)&deviceInput1);\nclerr= clSetKernelArg(clkern, 1, sizeof(cl_mem),(void *)&deviceInput2);\nclerr= clSetKernelArg(clkern, 2, sizeof(cl_mem),(void *)&deviceOutput);\nclerr= clSetKernelArg(clkern, 3, sizeof(int), &inputLength);\n wbTime_stop(Compute, \"Performing CUDA computation\");\n\n\nwbTime_start(Copy, \"Copying output memory to the CPU\");\n//@@ Copy the GPU memory back to the CPU here\ncl_event event=NULL;\nclerr= clEnqueueNDRangeKernel(clcmdq, clkern, 1, NULL, &globalSize, &localSize, 0, NULL, &event);\nclerr= clWaitForEvents(1, &event);\nclerr= clEnqueueReadBuffer(clcmdq, deviceOutput, CL_TRUE, 0,inputLength*sizeof(float), hostOutput, 0, NULL, NULL);\nwbTime_stop(Copy, \"Copying output memory to the CPU\");\n\n\nwbTime_start(GPU, \"Freeing GPU Memory\");\n//@@ Free the GPU memory here\nclReleaseMemObject(deviceInput1);\nclReleaseMemObject(deviceInput2);\nclReleaseMemObject(deviceOutput);\nwbTime_stop(GPU, \"Freeing GPU Memory\");\nwbSolution(args, hostOutput, inputLength);\nfree(hostInput1);\nfree(hostInput2);\nfree(hostOutput);\n\n  return 0;\n}"),
	}
	file6 := &embedded.EmbeddedFile{
		Filename:    `thrust_sample.cu`,
		FileModTime: time.Unix(1492675663, 0),
		Content:     string("#include <wb.h>\n#include <thrust/host_vector.h>\n#include <thrust/device_vector.h>\n\nint main(int argc, char *argv[]) {\n  wbArg_t args;\n  float *hostInput1;\n  float *hostInput2;\n  float *hostOutput;\n  int inputLength;\n\n  args = wbArg_read(argc, argv); /* parse the input arguments */\n\n  // Import host input data\n  wbTime_start(Generic, \"Importing data to host\");\n  hostInput1 =\n      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);\n  hostInput2 =\n      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);\n  wbTime_stop(Generic, \"Importing data to host\");\n\n  // Declare and allocate host output\n  //@@ Insert code here\n  hostOutput = (float*)malloc(sizeof(float) * inputLength);\n\n  wbTime_start(GPU, \"Doing GPU Computation (memory + compute)\");\n\n  // Declare and allocate thrust device input and output vectors\n  wbTime_start(GPU, \"Doing GPU memory allocation\");\n  //@@ Insert code here\n  thrust::device_vector<float> deviceInput1(inputLength);\n  thrust::device_vector<float> deviceInput2(inputLength);\n  thrust::device_vector<float> deviceOutput(inputLength);\n  wbTime_stop(GPU, \"Doing GPU memory allocation\");\n\n  // Copy to device\n  wbTime_start(Copy, \"Copying data to the GPU\");\n  //@@ Insert code here\n  thrust::copy(hostInput1, hostInput1 + inputLength, deviceInput1.begin());\n  thrust::copy(hostInput2, hostInput2 + inputLength, deviceInput2.begin());\n  wbTime_stop(Copy, \"Copying data to the GPU\");\n\n  // Execute vector addition\n  wbTime_start(Compute, \"Doing the computation on the GPU\");\n  //@@ Insert Code here\n  thrust::transform(deviceInput1.begin(), deviceInput1.end(),\n                    deviceInput2.begin(),\n                    deviceOutput.begin(),\n                    thrust::plus<float>());\n  wbTime_stop(Compute, \"Doing the computation on the GPU\");\n  /////////////////////////////////////////////////////////\n\n  // Copy data back to host\n  wbTime_start(Copy, \"Copying data from the GPU\");\n  //@@ Insert code here\n  thrust::copy(deviceOutput.begin(), deviceOutput.end(), hostOutput);\n  wbTime_stop(Copy, \"Copying data from the GPU\");\n\n  wbTime_stop(GPU, \"Doing GPU Computation (memory + compute)\");\n\n  wbSolution(args, hostOutput, inputLength);\n\n  free(hostInput1);\n  free(hostInput2);\n  free(hostOutput);\n  return 0;\n}"),
	}

	// define dirs
	dir1 := &embedded.EmbeddedDir{
		Filename:   ``,
		DirModTime: time.Unix(1492675663, 0),
		ChildFiles: []*embedded.EmbeddedFile{
			file2, // cpp_sample.cpp
			file3, // cuda_sample.cu
			file4, // openacc_sample.cpp
			file5, // opencl_sample.cpp
			file6, // thrust_sample.cu

		},
	}

	// link ChildDirs
	dir1.ChildDirs = []*embedded.EmbeddedDir{}

	// register embeddedBox
	embedded.RegisterEmbeddedBox(`_testcases`, &embedded.EmbeddedBox{
		Name: `_testcases`,
		Time: time.Unix(1492675663, 0),
		Dirs: map[string]*embedded.EmbeddedDir{
			"": dir1,
		},
		Files: map[string]*embedded.EmbeddedFile{
			"cpp_sample.cpp":     file2,
			"cuda_sample.cu":     file3,
			"openacc_sample.cpp": file4,
			"opencl_sample.cpp":  file5,
			"thrust_sample.cu":   file6,
		},
	})
}