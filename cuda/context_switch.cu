#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>

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