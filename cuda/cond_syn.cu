#include <stdio.h>
#include <cuda_runtime.h>
#include <bits/stdint-uintn.h>
#include <unistd.h>

#define THREADS_PER_GROUP 64
#define GROUP_NB 2

static char c;
static __device__ char devC;
static __device__ const int arrayNb = 2;

__global__ void cond_syn()
{
	/*该线程的ID*/
	int threadId = threadIdx.x;

	/*该线程所在组的ID*/
	int groupId = threadId / THREADS_PER_GROUP;

	/*该线程的相对ID*/
	int _threadId = threadId % THREADS_PER_GROUP;

	/*组的状态变量*/
	__shared__ int group[GROUP_NB];

	/*该组中已经完成任务的线程数量*/
	group[groupId] = THREADS_PER_GROUP;

	char array[arrayNb];
	array[0] = '-';
	array[1] = '*';

	for (int i = 0; i < 3; i++)
	{
		/*执行该线程的相关任务*/
		printf("%c%c groupId-%d-threadId-%d执行任务%d\n", array[0], array[1], groupId, threadId, i);

		/*组中最快的线程初始化组的状态变量为0*/
		atomicCAS((group + groupId), THREADS_PER_GROUP, 0);

		/*组中线程完成任务后更新组的状态变量*/
		int temp = atomicAdd((group + groupId), uint32_t(1));

		/*等待组中所有线程全部完成任务*/
		if (temp != THREADS_PER_GROUP - 1)
		{
			while (group[groupId] != THREADS_PER_GROUP)
			{
			}
		}

		if (_threadId == 0)
		{
			printf("%c groupId-%d完成任务%d\n", devC, groupId, i);
		}
	}
}

int main()
{
	char fileName[15] = "cond_syn.log";
	remove(fileName);
	int stdDup = dup(1);
	FILE *outLog = fopen(fileName, "a");
	dup2(fileno(outLog), 1);

	c = '$';
	cudaMemcpyToSymbol(devC, &c, sizeof(c));

	cond_syn<<<1, GROUP_NB * THREADS_PER_GROUP>>>();
	/*如果不加这句话main函数将不等cond_syn执行直接结束*/
	cudaDeviceSynchronize();

	fflush(stdout);
	fclose(outLog);
	dup2(stdDup, 1);
	close(stdDup);

	return 0;
}
/*
*vscode的工作目录必须为cuda*

rm -rf cond_syn cond_syn.o
/usr/local/cuda/bin/nvcc -ccbin g++ -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o cond_syn.o -c cond_syn.cu
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -g -G -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o cond_syn cond_syn.o -L /usr/local/cuda/lib64 -L /usr/local/cuda/samples/common/lib

./cond_syn

cuda-gdb
file cond_syn
r
q

rm -rf cond_syn cond_syn.o cond_syn.log
*/