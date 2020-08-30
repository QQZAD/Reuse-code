#include <stdio.h>
#include <string.h>
#include <unistd.h>

void checkWorkDir()
{
    char work[256];
    char file[256];
    char temp[256];

    getcwd(work, sizeof(work));
    int workLen = strlen(work);
    printf("当前工作目录%s\n", work);

    memcpy(file, __FILE__, sizeof(__FILE__));
    int fileLen = strlen(file);
    for (int i = fileLen - 1; i >= 0; i--)
    {
        if (file[i] == '/')
        {
            file[i] = '\0';
        }
    }
    fileLen = strlen(file);
    memcpy(file + workLen + 1, file, fileLen + 1);
    file[workLen] = '/';
    memcpy(file, work, workLen);
    printf("当前文件目录%s\n", file);

    printf("[切换工作目录]\n");
    chdir(file);

    /*执行该文件的相关工作*/

    printf("执行该文件的相关工作\n");
    getcwd(temp, sizeof(temp));
    printf("当前工作目录%s\n", temp);

    printf("[恢复工作目录]\n");
    chdir(work);
    getcwd(temp, sizeof(temp));
    printf("当前工作目录%s\n", temp);
}
