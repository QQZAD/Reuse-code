#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

#define nStreams 4
static cudaEvent_t startEvent, stopEvent;
static cudaStream_t stream[nStreams];

__global__ void kernel(float *a, int offset)
{
    int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
    float x = (float)i;
    float s = sinf(x);
    float c = cosf(x);
    a[i] += sqrtf(s * s + c * c);
}

void sequential(float *a, float *d_a, int bytes, int blockSize, int n)
{
    float ms;
    memset(a, 0, bytes);
    cudaEventRecord(startEvent, 0);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    kernel<<<n / blockSize, blockSize>>>(d_a, 0);
    cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("sequential数据传输和执行的总时间：%fms\n", ms);
}

void asynchronous1(float *a, float *d_a, int bytes, int blockSize, int streamSize, int streamBytes)
{
    float ms;
    memset(a, 0, bytes);
    cudaEventRecord(startEvent, 0);

    for (int i = 0; i < nStreams; i++)
    {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
        kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
        cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("asynchronous1数据传输和执行的总时间：%fms\n", ms);
}

void asynchronous2(float *a, float *d_a, int bytes, int blockSize, int streamSize, int streamBytes)
{
    float ms;
    memset(a, 0, bytes);
    cudaEventRecord(startEvent, 0);

    for (int i = 0; i < nStreams; i++)
    {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
    }
    for (int i = 0; i < nStreams; i++)
    {
        int offset = i * streamSize;
        kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    }
    for (int i = 0; i < nStreams; i++)
    {
        int offset = i * streamSize;
        cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("asynchronous2数据传输和执行的总时间：%fms\n", ms);
}

int main()
{
    int blockSize = 256;
    int n = 4 * 1024 * blockSize * nStreams;
    int streamSize = n / nStreams;
    int streamBytes = streamSize * sizeof(float);
    int bytes = n * sizeof(float);

    int devId = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devId);
    printf("是否支持执行与数据传输重叠：%d\n", prop.deviceOverlap);
    printf("异步引擎的数量：%d\n", prop.asyncEngineCount);
    cudaSetDevice(devId);

    /*分配设备内存和pinned主机内存*/
    float *a, *d_a;
    cudaMallocHost((void **)&a, bytes);
    cudaMalloc((void **)&d_a, bytes);

    /*创建开始事件和结束事件*/
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    /*创建Non-default流*/
    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    sequential(a, d_a, bytes, blockSize, n);
    asynchronous1(a, d_a, bytes, blockSize, streamSize, streamBytes);
    asynchronous2(a, d_a, bytes, blockSize, streamSize, streamBytes);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    for (int i = 0; i < nStreams; ++i)
    {
        cudaStreamDestroy(stream[i]);
    }
    cudaFree(d_a);
    cudaFreeHost(a);

    return 0;
}
/*
*vscode的工作目录必须为cuda*

rm -rf stream_asyn stream_asyn.o
/usr/local/cuda/bin/nvcc -ccbin g++ -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o stream_asyn.o -c stream_asyn.cu
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o stream_asyn stream_asyn.o -L /usr/local/cuda/lib64 -L /usr/local/cuda/samples/common/lib

./stream_asyn

cuda-gdb
file stream_asyn
r
q

rm -rf stream_asyn stream_asyn.o
*/