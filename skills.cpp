#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

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

    char fileName[10] = "debug.log";
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

int main()
{
    strCat(5);
    int number = 34;
    printf("%p\n", &number);
    redirectPrintf();
    return 0;
}
/*
g++ -g skills.cpp -o main;./main
rm -rf main debug.log
*/