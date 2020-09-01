#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <unistd.h>

#define ITEM_NB 4 //实际容量要减1
#define NEXT_ITEM(ID) ((ID + 1) % ITEM_NB)

static int *list;
static int flag[2] = {0};
static int *devList, *devFlag;

void *cpu_producer(void *argc)
{
    while (1)
    {
        if (flag[0] - flag[1] == 1)
        {
            printf("[cpu] 队列是满的\n");
        }
        while (flag[0] - flag[1] == 1)
        {
        }
        int task = rand() % 100 + 1;
        printf("[cpu] 在%d处插入任务%d\n", flag[1], task);
        // sleep(rand() % 5 + 1);
        list[flag[1]] = task;
        flag[1] = NEXT_ITEM(flag[1]);
        cudaMemcpy((void **)&devList, &list, ITEM_NB * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy((void **)&devFlag, &flag, 2 * sizeof(int), cudaMemcpyHostToDevice);
        break;
    }
    return NULL;
}

__global__ void gpu_kernel(int *devList, int *devFlag)
{
    /*该线程的ID*/
    int threadId = threadIdx.x;

    while (1)
    {
        if (threadId == 0 && devFlag[0] == devFlag[1])
        {
            printf("[gpu] 队列是空的\n");
        }
        while (devFlag[0] == devFlag[1])
        {
        }
        int id = devFlag[0];
        int task = devList[id];
        for (int i = 0; i < task; i++)
        {
            devList[id] *= (i + 1);
        }
        if (threadId == 0)
        {
            printf("[gpu] %d处的任务%d处理完成\n", id, task);
            devFlag[0] = NEXT_ITEM(devFlag[0]);
        }
    }
}

void *gpu_consumer(void *argc)
{
    gpu_kernel<<<1, 32>>>(devList, devFlag);
    /*如果不加这句话main函数将不等cond_syn执行直接结束*/
    cudaDeviceSynchronize();
    cudaMemcpy((void **)&list, &devList, ITEM_NB * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((void **)&flag, &devFlag, 2, cudaMemcpyDeviceToHost);
    return NULL;
}

int main()
{
    int len = ITEM_NB * sizeof(int);
    list = (int *)malloc(len);
    memset(list, 0, len);
    cudaMalloc((void **)&devList, len);
    cudaMalloc((void **)&devFlag, 2);
    cudaMemcpy((void **)&devList, &list, len, cudaMemcpyHostToDevice);
    cudaMemcpy((void **)&devFlag, &flag, 2, cudaMemcpyHostToDevice);

    pthread_t cpu_t, gpu_t;
    pthread_create(&cpu_t, NULL, cpu_producer, NULL);
    pthread_create(&gpu_t, NULL, gpu_consumer, NULL);
    pthread_join(cpu_t, NULL);
    pthread_join(gpu_t, NULL);

    free(list);
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