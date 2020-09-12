#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <bits/stdint-uintn.h>

#define PRINTF_TO_FILE

void newFolder(char *dirName)
{
    struct stat st = {0};
    if (stat(dirName, &st) == -1)
    {
        mkdir(dirName, 0700);
    }
}

void deleteFolder(char *dirName)
{
    struct stat st = {0};
    if (stat(dirName, &st) != -1)
    {
        rmdir(dirName);
    }
}

void deleteFile(char *fileName)
{
    FILE *file;
    if (file = fopen(fileName, "r"))
    {
        remove(fileName);
    }
}

void strCat(int number)
{
    char _str[8];
    sprintf(_str, "%02d", number);
    char str[15] = "name";
    int len = sizeof("name");
    mempcpy(_str + 2, ".txt", sizeof(".txt"));
    mempcpy(str + len - 1, _str, sizeof(_str));
    printf("%s\n", str);
}

void redirectPrintf()
{
    printf("[终端] 所有printf的输出信息输出到终端\n");

    char fileName[15] = "skills.log";
#ifdef PRINTF_TO_FILE
    remove(fileName);
    int stdDup = dup(1);
    FILE *outLog = fopen(fileName, "a");
    dup2(fileno(outLog), 1);
#endif

    printf("[文件] 所有printf的输出信息重定向到%s\n", fileName);

#ifdef PRINTF_TO_FILE
    fflush(stdout);
    fclose(outLog);
    dup2(stdDup, 1);
    close(stdDup);
#endif

    printf("[终端] 所有printf的输出信息恢复到终端\n");
}

int getRand(int a, int b)
{
    /*a,a+1,...,b-1,b*/
    return rand() % (b - a + 1) + a;
}

extern void checkWorkDir();
extern void backWorkDir();

struct alignas(4) stc
{
    /*
    默认2个字节对齐,必须是2的倍数
    结构体指针和uint8_t指针的转换受到字节对齐和结构体内变量定义先后顺序的影响
    */
    unsigned int a : 4; //占用4bit，来自字节对齐产生的多余空间
    unsigned int b : 4; //占用4bit，来自字节对齐产生的多余空间
    uint8_t c;
    uint8_t d;
    uint8_t e;
    uint16_t f;
    uint16_t g;
};

void x86_64()
{
    uint32_t a = 1;
    uint8_t *p = (uint8_t *)&a;
    printf("__LITTLE_ENDIAN 地址的低位存储值的低位\n%u-%u-%u-%u\n", p[0], p[1], p[2], p[3]);
}

int main()
{
    // strCat(5);
    // int number = 34;
    // printf("%p\n", &number);
    // redirectPrintf();
    // checkWorkDir();
    // backWorkDir();
    x86_64();
    return 0;
}
/*
cd cpp;g++ -c _skills/skills.cpp -o skills.o;g++ -g skills.cpp -o skills skills.o;./skills;cd ..
cd cpp;rm -rf skills.o skills skills.log;cd ..
*/