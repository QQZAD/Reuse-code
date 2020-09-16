#include <stdio.h>
#include <pthread.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <semaphore.h>
#include <atomic>

#define FLOWS_NB 4

static int id[FLOWS_NB];
static int *pac[FLOWS_NB] = {NULL};
static int nb[FLOWS_NB];
static int order[FLOWS_NB] = {3, 0, 2, 1};

static pthread_t input_t[FLOWS_NB];
/*在该实例中不用互斥锁也不会出现问题*/
// static pthread_mutex_t lock[FLOWS_NB];

static int shared = 0;

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

/*条件变量*/
static pthread_cond_t cond;

/*信号量*/
#define QUEUE_NB 5
static int queue[QUEUE_NB];
static sem_t psem, csem;

/*读写锁*/
static pthread_rwlock_t rwlock;

/*原子变量*/
std::atomic<int> atomInt;
template <typename BaseType>
struct atomic
{
    operator BaseType() const volatile;
};

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

/*多输入单输出*/
static void pthread()
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
    shared = 0;
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

void *producer(void *arg)
{
    int pos = 0;
    int num, count = 0;
    for (int i = 0; i < 12; i++)
    {
        num = rand() % 100;
        count += num;
        sem_wait(&psem);
        queue[pos] = num;
        sem_post(&csem);
        printf("producer: %d\n", num);
        pos = (pos + 1) % QUEUE_NB;
        sleep(rand() % 2);
    }
    printf("producer count=%d\n", count);
    return NULL;
}

void *consumer(void *arg)
{
    int pos = 0;
    int num, count = 0;
    for (int i = 0; i < 12; i++)
    {
        sem_wait(&csem);
        num = queue[pos];
        sem_post(&psem);
        printf("consumer: %d\n", num);
        count += num;
        pos = (pos + 1) % QUEUE_NB;
        sleep(rand() % 3);
    }
    printf("consumer count=%d\n", count);
    return NULL;
}

/*使用信号量*/
static void psemaphore()
{
    sem_init(&psem, 0, QUEUE_NB);
    sem_init(&csem, 0, 0);

    pthread_t tid[2];

    pthread_create(&tid[0], NULL, producer, NULL);
    pthread_create(&tid[1], NULL, consumer, NULL);

    pthread_join(tid[0], NULL);
    pthread_join(tid[1], NULL);

    sem_destroy(&psem);
    sem_destroy(&csem);
}

static void *pthreadWrite(void *arg)
{
    int i = (*(int *)arg);
    while (shared <= 100)
    {
        pthread_rwlock_wrlock(&rwlock);
        printf("write_thread_id=%d,shared=%d\n", i, shared += 20);
        pthread_rwlock_unlock(&rwlock);
        sleep(1);
    }
    return NULL;
}

static void *pthreadRead(void *arg)
{
    int i = (*(int *)arg);
    while (shared <= 100)
    {
        pthread_rwlock_rdlock(&rwlock);
        printf("read_thread_id=%d,shared=%d\n", i, shared);
        pthread_rwlock_unlock(&rwlock);
        sleep(1);
    }
    return NULL;
}

/*使用读写锁*/
static void prwlock()
{
    shared = 0;
    pthread_t thread1, thread2;
    int flag1 = 1;
    int flag2 = 2;

    pthread_rwlock_init(&rwlock, NULL);

    pthread_create(&thread1, NULL, pthreadWrite, (void *)&flag1);
    pthread_create(&thread2, NULL, pthreadRead, (void *)&flag2);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    pthread_rwlock_destroy(&rwlock);
}

static void *pthreadAtomic(void *arg)
{
    bool barg = *((bool *)arg);
    if (barg == true)
    {
        for (int i = 0; i < rand() % 20; i++)
        {
            atomInt++;
        }
    }
    else
    {
        for (int i = 0; i < rand() % 20; i++)
        {
            atomInt--;
        }
    }
    return NULL;
}

/*使用原子变量*/
static void patomic()
{
    atomInt = 0;
    bool arg[2] = {true, false};
    pthread_t pth[2];

    pthread_create(&pth[0], NULL, pthreadAtomic, (void *)arg);
    pthread_create(&pth[1], NULL, pthreadAtomic, (void *)(arg + 1));

    pthread_join(pth[0], NULL);
    pthread_join(pth[1], NULL);

    printf("atomInt=%d\n", int(atomInt));
}

int main()
{
    // pthread();
    // pmutex();
    // pcond();
    // psemaphore();
    // prwlock();
    patomic();
    return 0;
}

/*
cd cpp;g++ -g thread.cpp -o thread -lpthread;./thread;cd ..
cd cpp;rm -rf thread;cd ..
*/