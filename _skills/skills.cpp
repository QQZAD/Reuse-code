#include <stdio.h>
#include <string.h>
#include <unistd.h>

static char workDir[256];
static char fileDir[256];

/*检查工作目录*/
void checkWorkDir()
{
    getcwd(workDir, sizeof(workDir));
    int workLen = strlen(workDir);
    printf("[gpu schedule] 当前工作目录%s\n", workDir);

    memcpy(fileDir, __FILE__, sizeof(__FILE__));
    int fileLen = strlen(fileDir);
    bool isDir = false;
    for (int i = fileLen - 1; i >= 0; i--)
    {
        if (fileDir[i] == '/')
        {
            fileDir[i] = '\0';
            isDir = true;
            break;
        }
    }
    if (isDir == true)
    {
        fileLen = strlen(fileDir);
        memcpy(fileDir + workLen + 1, fileDir, fileLen + 1);
        fileDir[workLen] = '/';
    }
    memcpy(fileDir, workDir, workLen);
    printf("[gpu schedule] 当前文件目录%s\n", fileDir);

    if (strlen(fileDir) != strlen(workDir))
    {
        printf("[gpu schedule] 切换工作目录\n");
        chdir(fileDir);
    }
}

/*恢复工作目录*/
void backWorkDir()
{
    if (strlen(fileDir) != strlen(workDir))
    {
        printf("[gpu schedule] 恢复工作目录\n");
        chdir(workDir);
    }
}
