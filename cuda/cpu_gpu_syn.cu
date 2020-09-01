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
        bool temp = false;
        while (flag[0] - flag[1] == 1)
        {
            if (temp == false)
            {
                printf("[cpu] 队列是满的\n");
                temp = true;
            }
        }
        int task = rand() % 100 + 1;
        int id = flag[1];
        // sleep(rand() % 5 + 1);
        list[id] = task;
        flag[1] = NEXT_ITEM(flag[1]);
        cudaMemcpy(devList + id, list + id, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(devFlag + 1, flag + 1, sizeof(int), cudaMemcpyHostToDevice);

        printf("[cpu] 在%d处插入任务%d\n", id, task);
    }
    return NULL;
}

__global__ void gpu_kernel(int *devList, int *devFlag)
{
    /*该线程的ID*/
    int threadId = threadIdx.x;

    while (1)
    {
        // __syncthreads();
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
        // __syncthreads();
        if (threadId == 0)
        {
            printf("[gpu] %d处的任务%d处理完成\n", id, task);
            devFlag[0] = NEXT_ITEM(devFlag[0]);
        }
    }
}

int main()
{
    int len = ITEM_NB * sizeof(int);
    list = (int *)malloc(len);
    memset(list, 0, len);
    cudaMalloc((void **)&devList, len);
    cudaMalloc((void **)&devFlag, 2 * sizeof(int));
    cudaMemcpy(devList, list, len, cudaMemcpyHostToDevice);
    cudaMemcpy(devFlag, flag, 2 * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamDestroy(stream1);

    pthread_t cpu_t;
    pthread_create(&cpu_t, NULL, cpu_producer, NULL);
    gpu_kernel<<<1, 32>>>(devList, devFlag);

    pthread_join(cpu_t, NULL);
    cudaDeviceSynchronize();

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