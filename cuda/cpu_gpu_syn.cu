#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <unistd.h>

#define TASK_NB 4 //实际容量要减1
#define NEXT_ITEM(ID) ((ID + 1) % TASK_NB)

static int *list, *flag;
static int *devList, *devFlag;

static cudaStream_t streamHd;
static cudaStream_t streamDh;
static cudaStream_t streamKernel;

void *cpu_producer(void *argc)
{
    while (1)
    {
        bool temp = false;
        while (flag[0] == NEXT_ITEM(flag[1]))
        {
            if (temp == false)
            {
                printf("[cpu] 队列是满的\n");
                temp = true;
            }
        }
        int id = flag[1];
        int task = rand() % 100 + 1;
        list[id] = task;
        flag[1] = NEXT_ITEM(flag[1]);
        cudaMemcpyAsync(devList + id, list + id, sizeof(int), cudaMemcpyHostToDevice, streamHd);
        cudaMemcpyAsync(devFlag + 1, flag + 1, sizeof(int), cudaMemcpyHostToDevice, streamHd);
        printf("[cpu] 在%d处插入任务%d\n", id, task);
    }
    return NULL;
}

__global__ void gpu_consumer(int *devList, int *devFlag)
{
    int threadId = threadIdx.x;
    while (1)
    {
        __syncthreads();
        bool temp = false;
        while (devFlag[0] == devFlag[1])
        {
            if (threadId == 0)
            {
                if (temp == false)
                {
                    printf("[gpu] 队列是空的\n");
                    temp = true;
                }
            }
        }
        int id = devFlag[0];
        int task = devList[id];
        int result = 0;
        for (int i = 0; i < task; i++)
        {
            result += i * id * task;
        }
        __syncthreads();
        if (threadId == 0)
        {
            printf("[gpu] %d处的任务%d处理完成\n", id, task);
            devFlag[0] = NEXT_ITEM(devFlag[0]);
            cudaMemcpyAsync(list + id, devList + id, sizeof(int), cudaMemcpyDeviceToHost, streamDh);
            cudaMemcpyAsync(flag, devFlag, sizeof(int), cudaMemcpyDeviceToHost, streamDh);
        }
    }
}

void init()
{
    int listBytes = TASK_NB * sizeof(int);
    int flagBytes = 2 * sizeof(int);
    cudaMallocHost((void **)&list, listBytes);
    cudaMallocHost((void **)&flag, flagBytes);
    memset(list, 0, listBytes);
    memset(flag, 0, flagBytes);
    cudaMalloc((void **)&devList, listBytes);
    cudaMalloc((void **)&devFlag, flagBytes);
    cudaStreamCreate(&streamHd);
    cudaStreamCreate(&streamDh);
    cudaStreamCreate(&streamKernel);
    cudaMemcpyAsync(devList, list, listBytes, cudaMemcpyHostToDevice, streamHd);
    cudaMemcpyAsync(devFlag, flag, flagBytes, cudaMemcpyHostToDevice, streamHd);
}

void free()
{
    cudaStreamDestroy(streamHd);
    cudaStreamDestroy(streamDh);
    cudaStreamDestroy(streamKernel);
    cudaFree(devList);
    cudaFree(devFlag);
    cudaFreeHost(list);
    cudaFreeHost(flag);
}

int main()
{
    init();

    pthread_t cpu_t;
    pthread_create(&cpu_t, NULL, cpu_producer, NULL);
    gpu_consumer<<<1, 64, 0, streamKernel>>>(devList, devFlag);

    pthread_join(cpu_t, NULL);
    cudaDeviceSynchronize();

    free();
    return 0;
}
/*
*vscode的工作目录必须为cuda*

rm -rf cpu_gpu_syn cpu_gpu_syn.o
/usr/local/cuda/bin/nvcc -ccbin g++ -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o cpu_gpu_syn.o -c cpu_gpu_syn.cu
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o cpu_gpu_syn cpu_gpu_syn.o -L /usr/local/cuda/lib64 -L /usr/local/cuda/samples/common/lib

./cpu_gpu_syn

cuda-gdb
file cpu_gpu_syn
r
q

rm -rf cpu_gpu_syn cpu_gpu_syn.o
*/