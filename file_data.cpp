/*g++ -g file_data.cpp -o main;./main*/
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

int main()
{
    struct stat st = {0};
    if (stat("directory_name", &st) == -1)
    {
        mkdir("directory_name", 0700);
    }
    return 0;
}
/*rm -rf main directory_name*/