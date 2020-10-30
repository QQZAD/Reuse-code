// #include <iostream>
// #include <thread>
// #include <chrono>
// #include <future>
// #include <cmath>
// #include <vector>
// #include <cstdlib>
// using namespace std;

// double caculate(int v)
// {
//     if (v <= 0)
//     {
//         return v;
//     }
//     //假设这个计算很慢
//     this_thread::sleep_for(chrono::milliseconds(10));
//     return sqrt((v * v + sqrt((v - 5) * (v + 2.5)) / 2.0) / v);
// }

// template <typename Iter, typename Fun>
// double visitRange(thread::id id, Iter iterBegin, Iter iterEnd, Fun func)
// {
//     auto curId = this_thread::get_id();
//     if (id == this_thread::get_id())
//     {
//         cout << curId << " hello main thread\n";
//     }
//     else
//     {
//         cout << curId << " hello work thread\n";
//     }
//     double v = 0;
//     for (auto iter = iterBegin; iter != iterEnd; ++iter)
//     {
//         v += func(*iter);
//     }
//     return v;
// }

// int main()
// {
//     auto mainThreadId = std::this_thread::get_id();
//     //开启一个线程
//     std::vector<double> v;
//     for (int i = 0; i < 1000; i++)
//     {
//         v.push_back(rand());
//     }
//     cout << v.size() << endl;
//     double value = 0.0;
//     auto st = clock();
//     for (auto &info : v)
//     {
//         value += caculate(info);
//     }
//     auto ed = clock();
//     cout << "single thread: " << value << " " << ed - st << "time" << endl;

//     //下面用多线程来进行

//     auto iterMid = v.begin() + (v.size() / 2); // 指向整个vector一半部分
//     //计算后半部分
//     double anotherv = 0.0;
//     auto iterEnd = v.end();
//     st = clock();

//     thread s([&anotherv, mainThreadId, iterMid, iterEnd]() { // lambda
//         anotherv = visitRange(mainThreadId, iterMid, iterEnd, caculate);

//     });
//     // 计算前半部分
//     auto halfv = visitRange(mainThreadId, v.begin(), iterMid, caculate);

//     //关闭线程
//     s.join();

//     ed = clock();
//     cout << "multi thread: " << (halfv + anotherv) << " " << ed - st << "time" << endl;

//     getchar();
//     return 0;
// }

#include <iostream>
#include <thread>
#include <pthread.h>
#include <vector>
using namespace std;

#define CORE 2

int a[2][100];

struct Args
{
    long start;
    long size;
};

int sum = 0;
// void *thread_fun(void *argc)
// {
//     long start = ((Args *)argc)->start;
//     long size = ((Args *)argc)->size;
//     long temp = 0;
//     for (long i = start; i < size; i++)
//     {
//         for (int j = 0; j < 999; j++)
//         {
//             temp += j;
//         }
//         // cout << i << endl;
//     }
//     return NULL;
// }

void thread_fun(long start, long size)
{
    // long temp = 0;
    for (long i = start; i < size; i++)
    {
        // temp += i;
        // cout << i << endl;
    }
}

int main()
{
    long size = 80000000;
    long unit = size / CORE;

    Args par;
    par.start = 0;
    par.size = size;
    clock_t t1, t2;
    t1 = clock();
    // thread_fun(&par);
    // thread_fun(0, size);

    thread obj(thread_fun, 0, size);
    obj.join();
    t2 = clock();
    cout << t2 - t1 << endl;

    // pthread_t pth[CORE];
    // Args parg[CORE];

    // for (int i = 0; i < CORE; i++)
    // {
    //     parg[i].start = i * unit;
    //     parg[i].size = (i + 1) * unit - 1;
    // }
    // t1 = clock();
    // for (int i = 0; i < CORE; i++)
    // {
    //     pthread_create(&pth[i], NULL, thread_fun, (void *)(parg + i));
    // }

    // for (int i = 0; i < CORE; i++)
    // {
    //     pthread_join(pth[1], NULL);
    // }
    // t2 = clock();

    clock_t t3, t4;
    t3 = clock();
    thread obj1(thread_fun, 0, unit);
    thread obj2(thread_fun, unit, size);
    obj1.join();
    obj2.join();
    t4 = clock();

    // vector<thread> pth;
    // t1 = clock();
    // for (int i = 0; i < CORE; i++)
    // {
    //     thread obj(thread_fun, i * unit, unit * (i + 1));
    //     pth.push_back(obj);
    // }
    // for (int i = 0; i < CORE; i++)
    // {
    //     pth[i].join();
    // }
    //thread obj2(thread_fun, unit, size);
    // thread_fun(unit, size);
    // obj2.join();
    cout << t4 - t3 << endl;

    // getchar();
    return 0;
}

/*
cd cpp;g++ -g test.cpp -o test -lpthread;./test;cd ..
cd cpp;rm -rf test;cd ..
*/