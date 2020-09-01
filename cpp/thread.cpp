#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#define FLOWS_NB 4

static int id[FLOWS_NB];
static int *pac[FLOWS_NB] = {NULL};
static int nb[FLOWS_NB];
static int order[FLOWS_NB] = {3, 0, 2, 1};

static pthread_t input_t[FLOWS_NB];
/*在该实例中不用互斥锁也不会出现问题*/
// static pthread_mutex_t lock[FLOWS_NB];

void *flowInput(void *arg)
{
    int flowId = *(int *)arg;
    nb[flowId] = rand() % 10 + 1;
    printf("flowId-%d-nb-%d\n", flowId, nb[flowId]);
    int *temp = (int *)malloc(sizeof(int) * nb[flowId]);
    for (int i = 0; i < nb[flowId]; i++)
    {
        temp[i] = rand() % 50;
        sleep(1);
    }
    // pthread_mutex_lock(&lock[flowId]);
    pac[flowId] = temp;
    // pthread_mutex_unlock(&lock[flowId]);
    printf("flowId-%d-pac-%p\n", flowId, pac[flowId]);
}

void *input(void *arg)
{
    for (int i = 0; i < FLOWS_NB; i++)
    {
        /*这里不能直接传递共享变量i*/
        pthread_create(&(input_t[i]), NULL, flowInput, (void *)&(id[i]));
    }
}

void *process(void *arg)
{
    for (int i = 0; i < FLOWS_NB; i++)
    {
        int flowId = order[i];
        while (1)
        {
            // pthread_mutex_lock(&lock[flowId]);
            int *p = pac[flowId];
            // pthread_mutex_unlock(&lock[flowId]);
            if (p != NULL)
            {
                break;
            }
        }
        printf("flowId-%d处理完成\n", flowId);
    }
}

void freeMem()
{
    for (int i = 0; i < FLOWS_NB; i++)
    {
        pthread_join(input_t[i], NULL);
        // pthread_mutex_destroy(&(lock[i]));
        free(pac[i]);
    }
}

int main()
{
    for (int i = 0; i < FLOWS_NB; i++)
    {
        id[i] = i;
        // pthread_mutex_init(&(lock[i]), NULL);
    }

    pthread_t input_t, process_t;
    pthread_create(&input_t, NULL, input, NULL);
    pthread_create(&process_t, NULL, process, NULL);

    pthread_join(input_t, NULL);
    pthread_join(process_t, NULL);

    freeMem();
    return 0;
}
/*
cd cpp;g++ -g thread.cpp -o thread -lpthread;./thread;cd ..
cd cpp;rm -rf thread;cd ..
*/