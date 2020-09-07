#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>

/*
在CPU上，上下文切换是由内核中的一个名为“调度器”的函数在软件中完成的。
调度器是普通代码，是处理器必须运行的机器指令序列，而运行调度器所花费的时间是没有用于“有用”工作的时间。
一旦线程块在SM上启动，它的所有warp都将驻留，直到它们全部退出内核。GPU没有传统意义上的上下文切换。
SM更有可能从不同的warp而不是从相同的warp在一行中发出两条指令，如果不这样做，将使SM暴露于依赖暂停。

另一方面，GPU在硬件中进行上下文切换，而不需要调度器，而且它足够快。
当一个任务遇到"pipeline stall"时，另一个任务可以利用pipeline阶段，否则这些阶段将是空闲的。
这被称为“延迟隐藏”——一个任务的延迟被其他任务的进度所隐藏。
GPU使用上下文切换来隐藏延迟以获得更大的吞吐量。
*/

__global__ void gpu_consumer()
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

int main()
{
    gpu_consumer<<<1, WARP_SIZE, 0, streamKernel>>>();
    cudaDeviceSynchronize();
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