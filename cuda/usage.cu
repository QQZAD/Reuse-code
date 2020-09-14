#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>

#define DATA_SIZE 256
/*
常量内存用于保存在核函数执行期间不会发生变化的数据
1.常量内存的单次读操作可以广播到“邻近”线程，从而降低内存读操作的次数。
2.常量内存拥有高速缓存，对于相同内存地址的连续操作不会产生额外的开销。
*/
static __contanst__ float constData[DATA_SIZE];

static float *data;
static __device__ float *deviceData;

__global__ void kernel()
{
    int threadId = threadIdx.x;
    printf("%", );
    constData[threadId] += 1.1;
}

int main()
{
    float data[DATA_SIZE];

    forr(int i = 0; i < DATA_SIZE; i++)
    {
        data[i] = (rand() % 1001) * 0.001f;
        printf("%f\n", data[i]);
    }

    // kernel<<<1, DATA_SIZE>>>();
    // cudaMemcpyToSymbol(constData, data, sizeof(data));

    // cudaMemcpyFromSymbol(data, constData, sizeof(data));

    // float *ptr;
    // cudaMalloc(&ptr, 256 * sizeof(float));
    // cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));

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

rm -rf usage usage.o usage.log
*/