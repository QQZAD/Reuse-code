/*rm -rf main;g++ -g file_data.cpp -o main;./main*/
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

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

int main()
{
    strCat(5);
    return 0;
}