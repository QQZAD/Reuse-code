#include <iostream>
#include <thread>
#include <mutex>
#include <unistd.h>

using namespace std;

static int val = 1;
// 互斥锁
static mutex mt1, mt2;

static void fun1()
{
    printf("fun1 mt1.lock() start\n");
    mt1.lock();
    printf("fun1 mt1.lock() end\n");
    sleep(1);
    val = val + 1;
    printf("fun1 mt2.lock() start\n");
    mt2.lock();
    printf("fun1 mt2.lock() end\n");
    cout << val << endl;
    mt2.unlock();
    mt1.unlock();
}

static void fun2()
{
    printf("fun2 mt2.lock() start\n");
    mt2.lock();
    printf("fun2 mt2.lock() end\n");
    sleep(1);
    val = val * val;
    printf("fun2 mt1.lock() start\n");
    mt1.lock();
    printf("fun2 mt1.lock() end\n");
    cout << val << endl;
    mt1.unlock();
    mt2.unlock();
}

int main()
{
    thread t1(fun1), t2(fun2);
    t1.join();
    t2.join();
    cout << "OK" << endl;
    return 0;
}

/*
cd cpp;g++ -g -std=c++17 deadlock.cpp -o deadlock -lpthread;./deadlock;cd ..
cd cpp;rm -rf deadlock;cd ..
*/