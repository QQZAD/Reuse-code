#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*
嵌套函数定义
不符合C语言标准
gcc的扩展
*/
static void father()
{
    void son()
    {
        printf("Hello World\n");
    }
    son();
}

/*
将变量作为静态数组的长度

不能对数组进行初始化
并不是真正的变长
将数组的长度推迟到运行时确定
是否支持取决于编译器

数组不能是全局变量
int m = 3;
int a[m];
*/

static void special(int n)
{
    int b[n];
    for (int i = 0; i < n; i++)
    {
        b[i] = i;
    }
    printf("%ld\n", sizeof(b));
    printf("%p\n", b);
}

struct zeroBuffer
{
    int len;
    // 定义长度为0的数组实现变长结构体
    char data[0];
};

static void zeroBuffer()
{
    int currLen = 512;
    struct zeroBuffer *zBuffer = NULL;
    printf("%ld\n", sizeof(struct zeroBuffer));
    if ((zBuffer = (struct zeroBuffer *)malloc(sizeof(struct zeroBuffer) + currLen * sizeof(char))) != NULL)
    {
        zBuffer->len = currLen;
        memcpy(zBuffer->data, "Hello World", currLen);
        printf("%d, %s\n", zBuffer->len, zBuffer->data);
    }
    free(zBuffer);
    zBuffer = NULL;
}

int main()
{
    father();
    // special(5);
    // zeroBuffer();
}

/*
c11
c99
c90
c89
cd cpp;gcc -g -std=c11 c.c -o c;./c;cd ..
cd cpp;rm -rf c;cd ..
*/