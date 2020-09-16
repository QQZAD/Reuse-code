#include <stdio.h>
#include <pthread.h>
#include <asm-generic/errno.h>
#include <asm-generic/errno-base.h>
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

/*互斥锁*/
static pthread_mutex_t mutex;
enum mutexType
{
    NONE,
    NORMAL,
    TRY,
    TIMEOUT
};
static mutexType mt = NORMAL;

/*共享变量*/
static int shared = 0;

/*条件变量*/
static pthread_cond_t cond;

static void *flowInput(void *arg)
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

static void *input(void *arg)
{
    for (int i = 0; i < FLOWS_NB; i++)
    {
        /*这里不能直接传递共享变量i*/
        pthread_create(&(input_t[i]), NULL, flowInput, (void *)&(id[i]));
    }
}

static void *process(void *arg)
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

static void freeMem()
{
    for (int i = 0; i < FLOWS_NB; i++)
    {
        pthread_join(input_t[i], NULL);
        // pthread_mutex_destroy(&(lock[i]));
        free(pac[i]);
    }
}

static void thread()
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
}

static void *pmutexThread(void *arg)
{
    int flag = *((int *)arg);
    if (mt == NORMAL)
    {
        pthread_mutex_lock(&mutex);
    }
    else if (mt == TRY)
    {
        int err = pthread_mutex_trylock(&mutex);
        if (0 != err)
        {
            if (EBUSY == err)
            {
                printf("flag %d The mutex could not be acquired because it was already locked.\n", flag);
            }
        }
    }
    else if (mt == TIMEOUT)
    {
        struct timespec timeout;
        timeout.tv_sec = 2;
        timeout.tv_nsec = 0;
        int err = pthread_mutex_timedlock(&mutex, &timeout);
        if (0 != err)
        {
            if (ETIMEDOUT == err)
            {
                printf("flag %d The mutex could not be locked before the specified timeout expired.\n", flag);
            }
        }
    }
    for (int i = 97; i < 123; i++)
    {
        printf("%c%d", i, flag);
    }
    printf("\n");
    pthread_mutex_unlock(&mutex);
}

/*使用互斥锁*/
static void pmutex()
{
    pthread_t thread1, thread2;
    int flag1 = 1;
    int flag2 = 2;

    pthread_mutex_init(&mutex, NULL);

    pthread_create(&thread1, NULL, pmutexThread, (void *)&flag1);
    pthread_create(&thread2, NULL, pmutexThread, (void *)&flag2);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    pthread_mutex_destroy(&mutex);
}

static void *pcondThread(void *arg)
{
    pthread_mutex_lock(&mutex);
    pthread_cond_wait(&cond, &mutex);
    shared++;
    printf("Thread id is %ld ,shared=%d \n", pthread_self(), shared);
    pthread_mutex_unlock(&mutex);
    return NULL;
}

/*使用条件变量*/
static void pcond()
{
    pthread_t thread1, thread2;

    pthread_cond_init(&cond, NULL);
    pthread_mutex_init(&mutex, NULL);

    pthread_create(&thread1, NULL, pcondThread, NULL);
    pthread_create(&thread2, NULL, pcondThread, NULL);

    sleep(1);
    pthread_cond_signal(&cond);
    sleep(1);
    pthread_cond_signal(&cond);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    pthread_cond_destroy(&cond);
    pthread_mutex_destroy(&mutex);
}

int main()
{
    thread();
    return 0;
}
/*
cd cpp;g++ -g thread.cpp -o thread -lpthread;./thread;cd ..
cd cpp;rm -rf thread;cd ..
*/