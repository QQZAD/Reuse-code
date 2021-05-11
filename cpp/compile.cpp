#include <stdio.h>
#include <stdlib.h>
/*
#include "..."
当前头文件目录
编译器设定的头文件路径
系统变量CPLUS_INCLUDE_PATH或C_INCLUDE_PATH指定的头文件路径

#include <...>
编译器设置的头文件路径
系统变量CPLUS_INCLUDE_PATH或C_INCLUDE_PATH指定的头文件路径
*/

/*
*预处理：分析源代码的头文件和宏定义，生成预编译文件
*编译：生成预编译文件对应的汇编文件
*汇编：生成汇编文件对应的可重定向目标文件
*链接：将多个目标文件及库文件链接成可执行文件
*/

#include <unistd.h>
#include <sys/mman.h>

static void learnMalloc()
{
    int *p = (int *)malloc(sizeof(int) * 3);
    void *ADDR;
    // 分配虚拟地址空间
    // 将可访问数据空间的末尾(即“中断”)设置为ADDR
    // brk(ADDR);
    /*
    mmap将一块物理内存映射到多个进程的虚拟地址空间上
    来完成多个进程对同一块物理内存的读写
    映射地址从ADDR附近开始并扩展为LEN字节
    从偏移量到FD文件的描述根据PROT和标志
    如果ADDR不为零，它就是需要的映射地址
    如果在FLAGS中设置了MAP_FIXED位，则映射将精确地位于ADDR(必须是页面对齐的)
    否则系统会选择一个方便的就近地址
    返回值是选择的实际映射地址或错误时的MAP_FAILED(在这种情况下设置了' errno')
    成功的“mmap”调用会释放受影响区域的任何以前的映射
    */
    size_t LEN;
    int PROT;
    int FLAGS;
    int FD;
    off_t OFFSET;
    // mmap(ADDR,LEN,PROT,FLAGS,FD,OFFSET);
    /*
    为了减少内存碎片和系统调用开销，malloc采用内存池的方式，
    先申请大块内存作为堆区，然后将堆区分为多个内存块，以块作为内存管理的基本单元，
    当用户申请内存时，直接从堆区分配一块合适的空闲块，
    malloc采用隐式链表结构将堆区分成连续的、大小不一的块，包含已分配块和未分配块
    使用一个双向链表将空闲块连接起来，每一个空闲块记录了一个连续的、未分配的地址
    内存分配，malloc通过隐式链表遍历所有的空闲块，选择满足要求的块
    内存合并，malloc采用边界标记法，根据块的前后块是否分配来决定是否进行合并
    申请内存小于128K，使用系统函数brk在堆区分配
    申请内存大于128K，使用系统函数mmap在映射区分配 
    */
}

/*
C++的内存管理
堆区：手动分配的内存
栈区：函数的形式参数和局部变量
常量区：如字符串常量（无法修改）
静态数据区：全局变量和静态变量
代码区：二进制代码，共享、只读
*/

static void constString()
{
    char *const1 = (char *)"123";
    char const2[] = "123";
    // *const1 = '0';
    *const2 = '0';
    printf("%s\n", const2);
}

int main()
{
    constString();
}
/*
cd cpp;g++ -g -std=c++17 compile.cpp -o compile;./compile;cd ..
cd cpp;rm -rf compile;cd ..
*/