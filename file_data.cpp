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

void strCat(double number)
{
    char _str[8];
    sprintf(_str, "%.2lf", number);
    char str[20] = "abcd";
    int len = sizeof("abcd");
    mempcpy(str + len - 1, _str, sizeof(_str));
    printf("%s\n", str);
}

int main()
{
    strCat(123.56);
    return 0;
}