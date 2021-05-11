#include <stdio.h>

#define __IDENTIFIER(type, name) \
    type type##_##name

#define __GET_STRING(str) #str

typedef void *(*pfunc)(void *);

#define PFUNC(name) void *(*name)(void *)

#define __PRINTERROR(format, ...) \
    fprintf(stderr, format, __VA_ARGS__)

static void *pint(void *arg)
{
    printf("%d\n", *(int *)arg);
    return NULL;
}

static void *pfloat(void *arg)
{
    printf("%f\n", *(float *)arg);
    return NULL;
}

int main()
{
    __IDENTIFIER(int, a);
    __IDENTIFIER(float, b);
    int_a = 2;
    float_b = 3.6;
    printf("%d %f\n", int_a, float_b);
    printf("%s\n", __GET_STRING(hello));
    pfunc pi = pint;
    pi((void *)&int_a);
    PFUNC(pf);
    pf = pfloat;
    pf((void *)&float_b);
    __PRINTERROR("%s%d\n", "返回错误信息", -1);
}
/*
cd cpp;g++ -g -std=c++17 macro.cpp -o macro;./macro;cd ..
cd cpp;rm -rf macro;cd ..
*/