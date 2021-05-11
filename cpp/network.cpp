#include <sys/epoll.h>
#include <sys/socket.h>

// epoll是一种IO多路转接技术
// 在Linux网络编程中经常用来做事件触发
// 即当有特定事件到来时能够检测到
// 而不必阻塞进行监听
// epoll不需要遍历数组查询谁有事件
// epoll维护的是一棵红黑树
static void useEpoll()
{
    // 调用epoll_create()建立一个epoll对象
    // 在epoll文件系统中为这个句柄对象分配资源
    int epfd = epoll_create(32);
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);

    struct epoll_event ev, events[20];
    ev.data.fd = listenfd;
    ev.events = EPOLLIN | EPOLLET;

    // 调用epoll_ctl向epoll对象中添加连接的套接字
    epoll_ctl(epfd, EPOLL_CTL_ADD, listenfd, &ev);
    // 调用epoll_wait收集发生的事件的连接
    epoll_wait(epfd, events, 20, 500);
}

int main()
{
    useEpoll();
}

/*
cd cpp;g++ -g -std=c++17 network.cpp -o network;./network;cd ..
cd cpp;rm -rf network;cd ..
*/
