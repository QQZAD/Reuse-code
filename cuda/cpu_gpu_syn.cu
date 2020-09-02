#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <unistd.h>

#define TASK_NB 8
#define WARP_SIZE 32
#define LIST_SIZE 4 //实际容量要减1
#define NEXT_TASK(ID) ((ID + 1) % LIST_SIZE)

struct Task
{
    int id;
    int *pData;
    void set(int _id, int *_pData)
    {
        id = _id;
        pData = _pData;
    }
};

/*主机端内存*/
static struct Task *list;
static int *flag;

/*设备端访问主机端pinned内存*/
static struct Task *hostList;
static int *hostFlag;

/*主机端->设备端内存拷贝流*/
static cudaStream_t streamHd;

/*设备端->主机端内存拷贝流*/
static cudaStream_t streamDh;

/*内核执行流*/
static cudaStream_t streamKernel;

/*主机端生产者*/
void *cpu_producer(void *argc)
{
    for (int i = 1; i <= TASK_NB; i++)
    {
        bool temp = false;
        while (flag[0] == NEXT_TASK(flag[1]))
        {
            if (temp == false)
            {
                printf("[cpu] 队列是满的\n");
                temp = true;
            }
        }
        int id = flag[1];
        if (NEXT_TASK(id) != flag[0])
        {
            list[id].set(i);
            flag[1] = NEXT_TASK(flag[1]);

            int *data=(int*)


            cudaMemcpyAsync(devList + id, list + id, sizeof(struct Task), cudaMemcpyHostToDevice, streamHd);
            cudaMemcpyAsync(devFlag + 1, flag + 1, sizeof(int), cudaMemcpyHostToDevice, streamHd);
            printf("[cpu] 在%d处插入任务%d\n", id, i);
        }
    }
    return NULL;
}

/*设备端消费者*/
__global__ void gpu_consumer(struct Task *hostList, int *hostFlag)
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
        int task = devList[id].task[threadId];
        hostResult[id].task[threadId] = pow(task, 2);
        __syncthreads();
        if (threadId == 0)
        {
            printf("[gpu] %d处的任务%d处理完成\n", id, devList[id].taskId);
            devFlag[0] = NEXT_TASK(devFlag[0]);
            hostFlag[0] = devFlag[0];
        }
    }
}

/*保存结果到文件*/
void *cpu_saver(void *argc)
{
    remove("./result.txt");
    FILE *fp = fopen("./result.txt", "a+");
    while (1)
    {
        for (int id = 0; id < LIST_SIZE; id++)
        {
            if (list[id].isUsed == true)
            {
                fprintf(fp, "%d\t", list[id].taskId);
                for (int i = 0; i < WARP_SIZE; i++)
                {
                    fprintf(fp, "%d", result[id].task[i]);
                    if (i < WARP_SIZE - 1)
                    {
                        fprintf(fp, " ");
                    }
                }
                fprintf(fp, "\n");
                // list[id].isUsed = false;
            }
        }
    }
    fclose(fp);
    return NULL;
}

/*初始化*/
void init()
{
    int listBytes = LIST_SIZE * sizeof(struct Task);
    int flagBytes = 2 * sizeof(int);

    cudaMallocHost((void **)&list, listBytes, cudaHostAllocMapped);
    cudaMallocHost((void **)&flag, flagBytes, cudaHostAllocMapped);

    for (int i = 0; i < LIST_SIZE; i++)
    {
        list[i].set(0, NULL);
    }
    memset(flag, 0, flagBytes);

    cudaStreamCreate(&streamHd);
    cudaStreamCreate(&streamDh);
    cudaStreamCreate(&streamKernel);

    cudaHostGetDevicePointer<struct Task>(&hostList, (void *)list, 0);
    cudaHostGetDevicePointer<int>(&hostFlag, (void *)flag, 0);
}

/*清理*/
void free()
{
    cudaStreamDestroy(streamHd);
    cudaStreamDestroy(streamDh);
    cudaStreamDestroy(streamKernel);

    cudaFreeHost(list);
    cudaFreeHost(flag);
}

int main()
{
    init();

    pthread_t cpu_pro, cpu_sav;
    pthread_create(&cpu_sav, NULL, cpu_saver, NULL);
    gpu_consumer<<<1, WARP_SIZE, 0, streamKernel>>>(hostList, hostFlag);
    pthread_create(&cpu_pro, NULL, cpu_producer, NULL);

    pthread_join(cpu_pro, NULL);
    cudaDeviceSynchronize();
    pthread_join(cpu_sav, NULL);

    free();
    return 0;
}
/*
*vscode的工作目录必须为cuda*

rm -rf cpu_gpu_syn cpu_gpu_syn.o result.txt
/usr/local/cuda/bin/nvcc -ccbin g++ -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o cpu_gpu_syn.o -c cpu_gpu_syn.cu
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o cpu_gpu_syn cpu_gpu_syn.o -L /usr/local/cuda/lib64 -L /usr/local/cuda/samples/common/lib

./cpu_gpu_syn

cuda-gdb
file cpu_gpu_syn
r
q

rm -rf cpu_gpu_syn cpu_gpu_syn.o result.txt
*/