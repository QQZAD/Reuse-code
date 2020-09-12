#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum sortType
{
    ascend,
    descend
};

#define NB 100

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

int main()
{
    timespec start, end;
    long long a[NB];
    generate(a, NB);
    long long *b = (long long *)malloc(sizeof(long long) * NB);

    memcpy(b, a, sizeof(long long) * NB);
    printf("bubbleSort-start\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    bubbleSort(b, NB);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("bubbleSort-%lds\n", end.tv_sec - start.tv_sec);
    save((char *)"bubbleSort.txt", b, NB);

    memcpy(b, a, sizeof(long long) * NB);
    clock_gettime(CLOCK_MONOTONIC, &start);
    printf("quickSort-start\n");
    quickSort(b, 0, NB - 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("quickSort-%lds\n", end.tv_sec - start.tv_sec);
    save((char *)"quickSort.txt", b, NB);

    free(b);
    return 0;
}
/*
cd cpp;g++ -g sort.cpp -o sort;./sort;cd ..
cd cpp;rm -rf sort bubbleSort.txt quickSort.txt;cd ..
*/