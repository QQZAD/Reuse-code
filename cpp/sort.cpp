#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>

/*
算法的时间复杂度
基本操作的重复次数是关于问题规模的函数T(n)
引入辅助函数f(n)，lim(n->∞)(T(n)/f(n))=c>0
则O(f(n))为该算法的渐进时间复杂度

算法的空间复杂度
一般指算法用到的额外辅助空间的大小

Θ 精确渐近行为 f(n)=Θ(g(n)) ∃n0、c1、c2>0，∀n>=n0，0<=c1*g(n)<=f(n)<=c2*g(n)
O 上界 f(n)=O(g(n)) ∃n0、c>0，∀n>=n0，0<=f(n)<=c*g(n)
o 非紧上界 f(n)=o(g(n)) ∃n0、c>0，∀n>n0，0<=f(n)<c*g(n)
Ω 下界 f(n)=Ω(g(n)) ∃n0、c>0，∀n>=n0，0<=c*g(n)<=f(n)
ω 非紧下界 f(n)=ω(g(n)) ∃n0、c>0，∀n>n0，0<=c*g(n)<f(n)
*/

enum sortType
{
    ascend,
    descend
};

#define NB 100

timespec start, end;
long long a[NB];
long long *b = (long long *)malloc(sizeof(long long) * NB);

void generate(long long *array, long long nb)
{
    for (long long i = 0; i < nb; i++)
    {
        array[i] = rand() % (NB - (-NB) + 1) + (-NB);
    }
}

void save(char *filename, long long *array, long long nb)
{
    remove(filename);
    FILE *fp = fopen(filename, "w");
    for (long long i = 0; i < nb; i++)
    {
        fprintf(fp, "%lld\n", array[i]);
    }
    fclose(fp);
}

/*
时间复杂度：
最好Ω(n)
平均θ(n^2)
最坏n+n-1+n-2+...+2+1=n(n+1)/2 O(n^2)
空间复杂度：
在原数组实现排序
O(1)
*/
void bubbleSort(long long *array, long long nb, sortType st = ascend)
{
    for (long long i = 0; i < nb; i++)
    {
        for (long long j = i + 1; j < nb; j++)
        {
            bool judge;
            if (st == ascend)
            {
                judge = array[i] > array[j];
            }
            else if (st == descend)
            {
                judge = array[i] < array[j];
            }
            if (judge)
            {
                long long temp = array[j];
                array[j] = array[i];
                array[i] = temp;
            }
        }
    }
}

/*
时间复杂度：
最好Ω(n^2)
平均θ(n^2)
最坏O(n^2)
空间复杂度：
在原数组实现排序
O(1)
*/
void selectionSort(long long *array, long long nb, sortType st = ascend)
{
    for (int i = 0; i < nb - 1; i++)
    {
        int m = i;
        for (int j = i + 1; j < nb; j++)
        {
            bool judge;
            if (st == ascend)
            {
                judge = array[j] < array[m];
            }
            else if (st == descend)
            {
                judge = array[j] > array[m];
            }
            if (judge)
            {
                m = j;
            }
        }
        long long temp = array[m];
        array[m] = array[i];
        array[i] = temp;
    }
}

/*
时间复杂度：
最好Ω(nlog(2)n)=Ω(nlogn)
平均θ(nlog(2)n)=θ(nlogn)
最坏O(n^2)
空间复杂度：
O(logn)-O(n)
*/
void quickSort(long long *array, long long start, long long end, sortType st = ascend)
{
    if (start < end)
    {
        long long i = start;
        long long j = end;
        long long mark = array[start];
        while (i < j)
        {
            if (st == ascend)
            {
                while (i < j && array[j] >= mark)
                {
                    j--;
                }
                array[i] = array[j];
                while (i < j && array[i] <= mark)
                {
                    i++;
                }
                array[j] = array[i];
            }
            else if (st == descend)
            {
                while (i < j && array[j] <= mark)
                {
                    j--;
                }
                array[i] = array[j];
                while (i < j && array[i] >= mark)
                {
                    i++;
                }
                array[j] = array[i];
            }
        }
        array[i] = mark;
        quickSort(array, start, i, st);
        quickSort(array, i + 1, end, st);
    }
}

void bubble()
{
    memcpy(b, a, sizeof(long long) * NB);
    printf("bubbleSort-start\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    bubbleSort(b, NB);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("bubbleSort=%lfs\n", end.tv_sec - start.tv_sec + pow(10, -9) * (end.tv_nsec - start.tv_nsec));
    save((char *)"bubbleSort.txt", b, NB);
}

void selection()
{
    memcpy(b, a, sizeof(long long) * NB);
    printf("selectionSort-start\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    selectionSort(b, NB);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("selectionSort=%lfs\n", end.tv_sec - start.tv_sec + pow(10, -9) * (end.tv_nsec - start.tv_nsec));
    save((char *)"selectionSort.txt", b, NB);
}

void quick()
{
    memcpy(b, a, sizeof(long long) * NB);
    printf("quickSort-start\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    quickSort(b, 0, NB - 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("quickSort=%lfs\n", end.tv_sec - start.tv_sec + pow(10, -9) * (end.tv_nsec - start.tv_nsec));
    save((char *)"quickSort.txt", b, NB);
}

bool cmp(int a, int b)
{
    return a < b;
}

void lambda()
{
    std::vector<long long> vec(NB);
    for (int i = 0; i < vec.size(); i++)
    {
        vec[i] = rand() % (NB - 1 + 0);
    }
    std::vector<long long> lbvec(vec);
    sort(vec.begin(), vec.end(), cmp);

    printf("普通方式\n");
    for (long long it : vec)
    {
        std::cout << it << ' ';
    }
    printf("\n");

    sort(lbvec.begin(), lbvec.end(), [](int a, int b) -> bool { return a < b; });

    printf("lambda表达式\n");
    for (long long it : lbvec)
    {
        std::cout << it << ' ';
    }
    printf("\n");
}

int main()
{
    generate(a, NB);

    bubble();
    selection();
    quick();
    // lambda();

    free(b);
    return 0;
}
/*
cd cpp;g++ -g sort.cpp -o sort;./sort;cd ..
cd cpp;rm -rf sort bubbleSort.txt selectionSort.txt quickSort.txt;cd ..
*/