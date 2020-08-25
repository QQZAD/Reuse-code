/*rm -rf main;g++ -g file_data.cpp -o main;./main*/
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

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

int main()
{
    return 0;
}