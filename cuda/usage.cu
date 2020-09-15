#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <unistd.h>

#define DATA_SIZE 32
/*
每个SM上常量内存大小限制为64KB
1.常量内存的单次读操作可以广播到“邻近”线程，从而降低内存读操作的次数。
2.常量内存拥有高速缓存，对于相同内存地址的连续操作不会产生额外的开销。
*/
static __constant__ float constData[DATA_SIZE];
/*
纹理内存是只读的
*/
static texture<float> texData;

static float *_data;
/*
全局内存
*/
static __device__ float *deviceData;

float frand(int a, int b, int delta = 6)
{
    assert(b >= a);
    b *= pow(10, delta);
    a *= pow(10, delta);
    float d = pow(10, -delta);
    return (rand() % (b - a + 1) + a) * d;
}

__global__ void kernel()
{
    int threadId = threadIdx.x;
    printf("constData-threadId-%d-%f\n", threadId, constData[threadId]);
    printf("texData-threadId-%d-%f\n", threadId, tex1Dfetch(texData, threadId));
    printf("deviceData-threadId-%d-%f\n", threadId, deviceData[threadId]);
}

int main()
{
    int bytes = DATA_SIZE * sizeof(float);
    float data[DATA_SIZE];
    for (int i = 0; i < DATA_SIZE; i++)
    {
        data[i] = frand(10, 50);
        printf("START-%f\n", data[i]);
    }
    cudaMalloc((void **)&_data, bytes);
    cudaMemcpy(_data, data, bytes, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(constData, data, sizeof(data));

    cudaBindTexture(NULL, texData, _data, bytes);

    cudaMemcpyToSymbol(deviceData, &_data, sizeof(_data));

    kernel<<<1, DATA_SIZE>>>();

    cudaMemcpyFromSymbol(data, constData, sizeof(data));

    for (int i = 0; i < DATA_SIZE; i++)
    {
        printf("END-%f\n", data[i]);
    }

    cudaUnbindTexture(texData);
    cudaFree(_data);
    return 0;
}
/*
*vscode的工作目录必须为cuda*

rm -rf usage usage.o
/usr/local/cuda/bin/nvcc -ccbin g++ -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o usage.o -c usage.cu
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o usage usage.o -L /usr/local/cuda/lib64 -L /usr/local/cuda/samples/common/lib

./usage

cuda-gdb
file usage
r
q

rm -rf usage usage.o
*/