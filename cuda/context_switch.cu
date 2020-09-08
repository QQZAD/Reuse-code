/*
【warp上下文切换】微观
在CPU上，上下文切换是由内核中的一个名为“调度器”的函数在软件中完成的。
调度器是普通代码，是处理器必须运行的机器指令序列，而运行调度器所花费的时间是没有用于“有用”工作的时间。
一旦线程块在SM上启动，它的所有warp都将驻留，直到它们全部退出内核。GPU没有传统意义上的上下文切换。
SM更有可能从不同的warp而不是从相同的warp在一行中发出两条指令，如果不这样做，将使SM暴露于依赖暂停。

另一方面，GPU在硬件中进行上下文切换，而不需要调度器，而且它足够快。
当一个warp遇到"pipeline stall"时，另一个warp可以利用pipeline阶段，否则这些阶段将是空闲的。
这被称为“延迟隐藏”——一个warp的延迟被其他warp的进度所隐藏。
GPU使用上下文切换来隐藏延迟以获得更大的吞吐量。

【任务上下文切换】宏观
不同的kernel函数共享GPU上同一个SM，针对不同应用场景的三种抢占策略
1.Context switching：把一个SM上正在运行的thread block(TB)的上下文保存到内存，启动一个新的kernel函数抢占当前SM。
其切换开销对吞吐量影响！中！，对延迟影响！中！。
2.Draining：等待一个SM上正在运行的kernel函数的所有TB结束，启动一个新的kernel函数抢占当前SM。
其切换开销对吞吐量影响！小！，对延迟影响！大！。
3.Flushing：对于具有幂等性的kernel函数，即使强制结束当前正在运行的TB，重启后也不会对kernel函数的结果产生影响，不需要保存任何上下文信息。
其切换开销对吞吐量影响！大！（当抢占发生在任务即将结束时），对延迟影响！很小！。

幂等性：在编程中一个幂等操作的特点是其任意多次执行所产生的影响均与一次执行的影响相同。
使用相同参数重复执行能获得相同结果。不会影响系统状态，也不用担心重复执行会对系统造成改变。
*/
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <pthread.h>

#define MAX_SIZE 999999
#define NUM_OF_TASK 2
#define TASK1 0
#define TASK2 1

#define GET_TASK(TASK_ID) (TASK_ID == TASK1 ? task1 : task2)

static int NUM_OF_SM = 0;
static int THREADS_PER_BLOCK = 0;
static int BLOCKS_PER_SM = 0;

/*
1表示特定任务在特定SM上执行
0表示特定任务没有在特定SM上执行
-1表示特定SM上的特定任务将被抢占
*/
static int *state;
static int *hostState;

// static int *lastTask;
// static int *hostLastTask;

void (*ptask)(volatile int *hostState, int smId, int nbSm);

void init(int gpu)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpu);
    NUM_OF_SM = deviceProp.multiProcessorCount;
    printf("NUM_OF_SM-%d\n", NUM_OF_SM);
    THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
    printf("THREADS_PER_BLOCK-%d\n", THREADS_PER_BLOCK);
    BLOCKS_PER_SM = deviceProp.maxThreadsPerMultiProcessor / THREADS_PER_BLOCK;
    printf("BLOCKS_PER_SM-%d\n", BLOCKS_PER_SM);

    int bytes = sizeof(int) * NUM_OF_TASK * NUM_OF_SM;
    cudaMallocHost((void **)&state, bytes, cudaHostAllocMapped);
    // cudaMallocHost((void **)&lastTask, bytes, cudaHostAllocMapped);
    memset(state, 0, bytes);
    // for (int i = 0; i < bytes / sizeof(int); i++)
    // {
    //     lastTask[i] = -1;
    // }
    cudaHostGetDevicePointer<int>(&hostState, (void *)state, 0);
    // cudaHostGetDevicePointer<int>(&hostLastTask, (void *)lastTask, 0);
}

void free()
{
    cudaFreeHost(state);
}

static __device__ __inline__ int getSmid()
{
    int smId;
    asm volatile("mov.u32 %0, %%smid;"
                 : "=r"(smId));
    return smId;
}

__global__ void task1(volatile int *hostState, int smId, int nbSm)
{
    if (smId == getSmid())
    {
        int threadId = threadIdx.x;
        int instanceId = TASK1 * nbSm + smId;
        if (threadId == 0)
        {
            printf("正在执行任务1\n");
        }
        for (int i = 0; i < MAX_SIZE; i++)
        {
            if (hostState[instanceId] == -1)
            {
                /*执行返回时需要同步所有线程*/
                return;
            }
            else
            {
                /*执行任务1*/
            }
        }
        if (threadId == 0)
        {
            printf("任务1执行完成\n");
        }
        hostState[instanceId] = 0;
    }
}

__global__ void task2(volatile int *hostState, int smId, int nbSm)
{
    if (smId == getSmid())
    {
        int threadId = threadIdx.x;
        int instanceId = TASK2 * nbSm + smId;
        if (threadId == 0)
        {
            printf("正在执行任务2\n");
        }
        for (int i = 0; i < MAX_SIZE; i++)
        {
            if (hostState[instanceId] == -1)
            {
                /*执行返回时需要同步所有线程*/
                return;
            }
            else
            {
                /*执行任务2*/
            }
        }
        if (threadId == 0)
        {
            printf("任务2执行完成\n");
        }
        hostState[instanceId] = 0;
    }
}

void contextSwitch(int smId, int taskId)
{
}

void draining(int smId, int taskId)
{
    assert(smId < NUM_OF_SM);
    int instanceId = taskId * NUM_OF_SM + smId;
    for (int i = 0; i < NUM_OF_TASK; i++)
    {
        while (hostState[i * NUM_OF_SM + smId] != 0)
        {
        }
    }
    hostState[instanceId] = 1;
    ptask = GET_TASK(taskId);
    ptask<<<NUM_OF_SM, THREADS_PER_BLOCK>>>(hostState, smId, NUM_OF_SM);
}

void *flushingTail(void *argc)
{
    int lastTask = *((int *)argc);
    while (hostState[instanceId] != 0)
    {
    }
    hostState[lastTask * NUM_OF_SM + smId] = 1;
    ptask = GET_TASK(lastTask);
    ptask<<<NUM_OF_SM, THREADS_PER_BLOCK>>>(hostState, smId, NUM_OF_SM);
    return NULL;
}

void flushing(int smId, int taskId)
{
    assert(smId < NUM_OF_SM);
    int lastTask = -1;
    int instanceId = taskId * NUM_OF_SM + smId;
    pthread_t tail;
    for (int i = 0; i < NUM_OF_TASK; i++)
    {
        if (i != taskId && hostState[i * NUM_OF_SM + smId] == 1)
        {
            lastTask = i;
            hostState[i * NUM_OF_SM + smId] = -1;
        }
    }
    hostState[instanceId] = 1;
    ptask = GET_TASK(taskId);
    ptask<<<NUM_OF_SM, THREADS_PER_BLOCK>>>(hostState, smId, NUM_OF_SM);
    if (lastTask != -1)
    {
        pthread_create(&tail, NULL, flushingTail, NULL);
        // while (hostState[instanceId] != 0)
        // {
        // }
        // hostState[lastTask * NUM_OF_SM + smId] = 1;
        // ptask = GET_TASK(lastTask);
        // ptask<<<NUM_OF_SM, THREADS_PER_BLOCK>>>(hostState, smId, NUM_OF_SM);
    }
}

int main()
{
    init(0);
    int smId = 1;
    flushing(smId, TASK1);
    sleep(1);
    flushing(smId, TASK2);
    sleep(1);
    flushing(smId, TASK1);
    cudaDeviceSynchronize();
    free();
    return 0;
}
/*
*vscode的工作目录必须为cuda*

rm -rf context_switch context_switch.o
/usr/local/cuda/bin/nvcc -ccbin g++ -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o context_switch.o -c context_switch.cu
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o context_switch context_switch.o -L /usr/local/cuda/lib64 -L /usr/local/cuda/samples/common/lib

./context_switch

cuda-gdb
file context_switch
r
q

rm -rf context_switch context_switch.o
*/