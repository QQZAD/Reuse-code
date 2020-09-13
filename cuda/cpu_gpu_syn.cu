#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <unistd.h>

#define TASK_NB 100
#define WARP_SIZE 32
#define LIST_SIZE 6 //实际容量要减1
#define NEXT_TASK(ID) ((ID + 1) % LIST_SIZE)

struct Task
{
    int id;
    int nb;
    int *pData;
    int *pDevResult;
    int *pHostResult;
    bool isSave;
    Task()
    {
        id = 0;
        nb = 0;
        pData = NULL;
        pDevResult = NULL;
        pHostResult = NULL;
        isSave = false;
    }
};

/*主机端内存*/
static struct Task *list;
static int *flag;
static int *finTaksNb;

/*设备端访问主机端pinned内存*/
static struct Task *devList;
static int *devFlag;
static int *devFinTaksNb;

/*主机端->设备端内存拷贝流*/
static cudaStream_t streamHd;

/*设备端->主机端内存拷贝流*/
static cudaStream_t streamDh;

/*内核执行流*/
static cudaStream_t streamKernel;

/*CUDA异常处理*/
static cudaError err;

/*主机端生产者*/
void *cpuProducer(void *argc)
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
        int cur = flag[1];
        list[cur].nb = rand() % (WARP_SIZE - 1 + 1) + 1;
        int bytes = sizeof(int) * list[cur].nb;
        int *data = (int *)malloc(bytes);
        for (int j = 0; j < list[cur].nb; j++)
        {
            data[j] = i;
        }
        err = cudaMemcpyAsync(list[cur].pData, data, bytes, cudaMemcpyHostToDevice, streamHd);
        if (err != 0)
        {
            printf("[cudaError] cudaMemcpyAsync返回0x%x\n", err);
            exit(1);
        }
        err = cudaMemcpyAsync(list[cur].pDevResult, list[cur].pHostResult, bytes, cudaMemcpyHostToDevice, streamHd);
        if (err != 0)
        {
            printf("[cudaError] cudaMemcpyAsync返回0x%x\n", err);
            exit(1);
        }
        cudaStreamSynchronize(streamHd);
        list[cur].id = i;
        flag[1] = NEXT_TASK(cur);
        free(data);
        printf("[cpu] 在%d处插入任务%d\n", cur, i);
    }
    return NULL;
}

/*设备端消费者*/
__global__ void gpuConsumer(struct Task *devList, int *devFlag, int *devFinTaksNb)
{
    int threadId = threadIdx.x;
    while (devFinTaksNb[0] != TASK_NB)
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
        int cur = devFlag[0];
        if (threadId < devList[cur].nb)
        {
            int task = devList[cur].pData[threadId];
            devList[cur].pDevResult[threadId] = pow(task, 2) - task;
        }
        __syncthreads();
        if (threadId == 0)
        {
            printf("[gpu] %d处的任务%d处理完成\n", cur, devList[cur].id);
            devList[cur].isSave = true;
            while (devList[cur].isSave == true)
            {
            }
        }
    }
}

/*保存结果到文件*/
void *cpuSaver(void *argc)
{
    while (finTaksNb[0] != TASK_NB)
    {
        int cur = flag[0];
        if (list[cur].nb > 0)
        {
            while (list[cur].isSave == false)
            {
            }
            int bytes = sizeof(int) * list[cur].nb;
            err = cudaMemcpyAsync(list[cur].pHostResult, list[cur].pDevResult, bytes, cudaMemcpyDeviceToHost, streamDh);
            if (err != 0)
            {
                printf("[cudaError] cudaMemcpyAsync返回0x%x\n", err);
                exit(1);
            }
            cudaStreamSynchronize(streamDh);
            FILE *fp = fopen("./result.txt", "a+");
            fprintf(fp, "%d\t", list[cur].id);
            for (int i = 0; i < list[cur].nb; i++)
            {
                fprintf(fp, "%d", list[cur].pHostResult[i]);
                if (i < list[cur].nb - 1)
                {
                    fprintf(fp, " ");
                }
            }
            fprintf(fp, "\n");
            fclose(fp);
            printf("[cpu] %d处的任务%d结果已经保存\n", cur, list[cur].id);
            list[cur].isSave = false;
            flag[0] = NEXT_TASK(cur);
            (finTaksNb[0])++;
        }
    }
    return NULL;
}

/*初始化*/
void init()
{
    remove("./result.txt");
    int listBytes = LIST_SIZE * sizeof(struct Task);
    int flagBytes = 2 * sizeof(int);

    cudaMallocHost((void **)&list, listBytes, cudaHostAllocMapped);
    cudaMallocHost((void **)&flag, flagBytes, cudaHostAllocMapped);
    cudaMallocHost((void **)&finTaksNb, sizeof(int), cudaHostAllocMapped);
    memset(flag, 0, flagBytes);
    memset(finTaksNb, 0, sizeof(int));

    for (int i = 0; i < LIST_SIZE; i++)
    {
        err = cudaMalloc((void **)&(list[i].pData), sizeof(int) * WARP_SIZE);
        if (err != 0)
        {
            printf("[cudaError] cudaMalloc返回0x%x\n", err);
            exit(1);
        }
        err = cudaMalloc((void **)&(list[i].pDevResult), sizeof(int) * WARP_SIZE);
        if (err != 0)
        {
            printf("[cudaError] cudaMalloc返回0x%x\n", err);
            exit(1);
        }
        list[i].pHostResult = (int *)malloc(sizeof(int) * WARP_SIZE);
        memset(list[i].pHostResult, 0, sizeof(int) * WARP_SIZE);
    }

    cudaStreamCreate(&streamHd);
    cudaStreamCreate(&streamDh);
    cudaStreamCreate(&streamKernel);

    cudaHostGetDevicePointer<struct Task>(&devList, (void *)list, 0);
    cudaHostGetDevicePointer<int>(&devFlag, (void *)flag, 0);
    cudaHostGetDevicePointer<int>(&devFinTaksNb, (void *)finTaksNb, 0);
}

/*清理*/
void free()
{
    cudaStreamDestroy(streamHd);
    cudaStreamDestroy(streamDh);
    cudaStreamDestroy(streamKernel);

    for (int i = 0; i < LIST_SIZE; i++)
    {
        cudaFree(list[i].pData);
        cudaFree(list[i].pDevResult);
        free(list[i].pHostResult);
    }

    cudaFreeHost(list);
    cudaFreeHost(flag);
    cudaFreeHost(finTaksNb);
}

int main()
{
    init();

    pthread_t cpu_pro, cpu_sav;
    pthread_create(&cpu_sav, NULL, cpuSaver, NULL);
    gpuConsumer<<<1, WARP_SIZE, 0, streamKernel>>>(devList, devFlag, devFinTaksNb);
    pthread_create(&cpu_pro, NULL, cpuProducer, NULL);

    pthread_join(cpu_pro, NULL);
    printf("cpuProducer已经退出\n");
    cudaDeviceSynchronize();
    printf("gpuConsumer已经退出\n");
    pthread_join(cpu_sav, NULL);
    printf("cpuSaver已经退出\n");

    free();
    return 0;
}

/*
*vscode的工作目录必须为cuda*
rm -rf cpu_gpu_syn cpu_gpu_syn.o result.txt
/usr/local/cuda/bin/nvcc -ccbin g++ -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o cpu_gpu_syn.o -c cpu_gpu_syn.cu -dc
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o cpu_gpu_syn cpu_gpu_syn.o -L /usr/local/cuda/lib64 -L /usr/local/cuda/samples/common/lib

./cpu_gpu_syn

cuda-gdb
file cpu_gpu_syn
r
q

rm -rf cpu_gpu_syn cpu_gpu_syn.o result.txt
*/