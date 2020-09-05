#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <unistd.h>

#define TASK_NB 100
#define WARP_SIZE 32
#define LIST_SIZE 21 //实际容量要减1
#define NEXT_TASK(ID) ((ID + 1) % LIST_SIZE)

struct Task
{
    int id;
    int *pData;
    int *pResult;
    bool isSave;
    Task()
    {
        id = 0;
        pData = NULL;
        pResult = NULL;
        isSave = false;
    }
};

/*主机端内存*/
static struct Task *list;
static int *flag;
static int *finTaksNb;

/*设备端访问主机端pinned内存*/
static struct Task *hostList;
static int *hostFlag;
static int *hostFinTaksNb;

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
        int cur = flag[1];
        int bytes = sizeof(int) * WARP_SIZE;
        int *data = (int *)malloc(bytes);
        int *result = (int *)malloc(bytes);
        for (int j = 0; j < WARP_SIZE; j++)
        {
            data[j] = i;
            result[j] = 0;
        }
        cudaMalloc((void **)&(list[cur].pData), bytes);
        cudaMalloc((void **)&(list[cur].pResult), bytes);
        cudaMemcpyAsync(list[cur].pData, data, bytes, cudaMemcpyHostToDevice, streamHd);
        cudaMemcpyAsync(list[cur].pResult, result, bytes, cudaMemcpyHostToDevice, streamHd);
        list[cur].id = i;
        flag[1] = NEXT_TASK(cur);
        free(data);
        free(result);
        printf("[cpu] 在%d处插入任务%d\n", cur, i);
    }
    return NULL;
}

/*设备端消费者*/
__global__ void gpu_consumer(struct Task *hostList, int *hostFlag, int *hostFinTaksNb)
{
    int threadId = threadIdx.x;
    while (hostFinTaksNb[0] != TASK_NB)
    {
        __syncthreads();
        bool temp = false;
        while (hostFlag[0] == hostFlag[1])
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
        int cur = hostFlag[0];
        int task = hostList[cur].pData[threadId];
        hostList[cur].pResult[threadId] = pow(task, 2) - task;
        __syncthreads();
        if (threadId == 0)
        {
            printf("[gpu] %d处的任务%d处理完成\n", cur, hostList[cur].id);
            hostList[cur].isSave = true;
            while (hostList[cur].isSave == true)
            {
            }
        }
    }
}

/*保存结果到文件*/
void *cpu_saver(void *argc)
{
    while (finTaksNb[0] != TASK_NB)
    {
        int cur = flag[0];
        while (list[cur].isSave == false)
        {
        }
        int resultBytes = sizeof(int) * WARP_SIZE;
        int *result = (int *)malloc(resultBytes);
        cudaMemcpyAsync(result, list[cur].pResult, resultBytes, cudaMemcpyDeviceToHost, streamDh);
        FILE *fp = fopen("./result.txt", "a+");
        fprintf(fp, "%d\t", list[cur].id);
        for (int i = 0; i < WARP_SIZE; i++)
        {
            fprintf(fp, "%d", result[i]);
            if (i < WARP_SIZE - 1)
            {
                fprintf(fp, " ");
            }
        }
        fprintf(fp, "\n");
        fclose(fp);
        free(result);
        flag[0] = NEXT_TASK(cur);
        list[cur].isSave = false;
        (finTaksNb[0])++;
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

    cudaStreamCreate(&streamHd);
    cudaStreamCreate(&streamDh);
    cudaStreamCreate(&streamKernel);

    cudaHostGetDevicePointer<struct Task>(&hostList, (void *)list, 0);
    cudaHostGetDevicePointer<int>(&hostFlag, (void *)flag, 0);
    cudaHostGetDevicePointer<int>(&hostFinTaksNb, (void *)finTaksNb, 0);
}

/*清理*/
void free()
{
    cudaStreamDestroy(streamHd);
    cudaStreamDestroy(streamDh);
    cudaStreamDestroy(streamKernel);

    cudaFreeHost(list);
    cudaFreeHost(flag);
    cudaFreeHost(finTaksNb);
}

int main()
{
    init();

    pthread_t cpu_pro, cpu_sav;
    pthread_create(&cpu_sav, NULL, cpu_saver, NULL);
    gpu_consumer<<<1, WARP_SIZE, 0, streamKernel>>>(hostList, hostFlag, hostFinTaksNb);
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