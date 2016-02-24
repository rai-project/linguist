#include <wb.h> //@@ wb include opencl.h for you
#include <math.h>
//@@ OpenCL Kernel
const char* vaddsrc ="__kernel void vadd(__global const float *a,__global const float *b,__global float *result){int id = get_global_id(0);result[id] = a[id] + b[id];}";

int main(int argc, char **argv)
{

    unsigned int VECTOR_SIZE = 1024;
    int size = VECTOR_SIZE* sizeof(float);
    wbArg_t args;
    int inputLength = VECTOR_SIZE;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;
    cl_mem deviceInput1;
    cl_mem deviceInput2;
    cl_mem deviceOutput;

  args = wbArg_read(argc, argv);
  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
   wbTime_stop(Generic, "Importing data and creating memory on host");
  wbLog(TRACE, "The input length is ", inputLength);


  //@@ OpenCL Context Setup Code (simple)
        size_t parmsz;
        cl_int clerr;
        cl_context clctx;
        cl_command_queue clcmdq;
        cl_program clpgm;
        cl_kernel clkern;

// query the number of platforms
   cl_uint numPlatforms;
   clerr = clGetPlatformIDs(0, NULL, &numPlatforms);
   cl_platform_id platforms[numPlatforms];
   clerr = clGetPlatformIDs(numPlatforms, platforms, NULL);
   cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (unsigned long)platforms[0], 0};
   clctx = clCreateContextFromType(properties,CL_DEVICE_TYPE_ALL, NULL, NULL, &clerr);
    clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0,NULL, & parmsz);
   cl_device_id* cldevs = (cl_device_id *) malloc( parmsz);
   clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz,cldevs, NULL);
   clcmdq = clCreateCommandQueue(clctx,cldevs[0], 0, &clerr);
   clpgm = clCreateProgramWithSource(clctx, 1, &vaddsrc,NULL, &clerr);
   char clcompileflags[4096];
   sprintf(clcompileflags, "-cl-mad-enable");
   clerr = clBuildProgram(clpgm, 0, NULL, clcompileflags,NULL, NULL);
   clkern = clCreateKernel(clpgm, "vadd", &clerr);
  //@@ OpenCL Context Setup Code (simple)


  wbTime_start(GPU, "Allocating GPU memory.Copying input memory to the GPU.");
  //@@ Allocate GPU memory here Copy memory to the GPU here
  deviceInput1 = clCreateBuffer(clctx, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR,inputLength *sizeof(float), hostInput1, NULL);
  deviceInput2 = clCreateBuffer(clctx, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR, inputLength *sizeof(float), hostInput2, NULL);
  deviceOutput = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY,inputLength *sizeof(float), NULL, NULL);
  wbTime_stop(GPU, "Allocating GPU memory.Copying input memory to the GPU.");



  //@@ Initialize the grid and block dimensions here
size_t globalSize, localSize;
localSize = 64;
globalSize = ceil(inputLength/(float)localSize)*localSize;




  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
clerr= clSetKernelArg(clkern, 0, sizeof(cl_mem),(void *)&deviceInput1);
clerr= clSetKernelArg(clkern, 1, sizeof(cl_mem),(void *)&deviceInput2);
clerr= clSetKernelArg(clkern, 2, sizeof(cl_mem),(void *)&deviceOutput);
clerr= clSetKernelArg(clkern, 3, sizeof(int), &inputLength);
 wbTime_stop(Compute, "Performing CUDA computation");


wbTime_start(Copy, "Copying output memory to the CPU");
//@@ Copy the GPU memory back to the CPU here
cl_event event=NULL;
clerr= clEnqueueNDRangeKernel(clcmdq, clkern, 1, NULL, &globalSize, &localSize, 0, NULL, &event);
clerr= clWaitForEvents(1, &event);
clerr= clEnqueueReadBuffer(clcmdq, deviceOutput, CL_TRUE, 0,inputLength*sizeof(float), hostOutput, 0, NULL, NULL);
wbTime_stop(Copy, "Copying output memory to the CPU");


wbTime_start(GPU, "Freeing GPU Memory");
//@@ Free the GPU memory here
clReleaseMemObject(deviceInput1);
clReleaseMemObject(deviceInput2);
clReleaseMemObject(deviceOutput);
wbTime_stop(GPU, "Freeing GPU Memory");
wbSolution(args, hostOutput, inputLength);
free(hostInput1);
free(hostInput2);
free(hostOutput);

  return 0;
}