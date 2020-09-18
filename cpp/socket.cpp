#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>

#define PORT 23   //端口号
#define BACKLOG 5 //最大监听数

int main()
{
    /*socketFd表示socket文件描述符*/
    int socketFd = 0;
    int iRecvLen = 0;                   //接收成功后的返回值
    int new_fd = 0;                     //建立连接后的句柄
    char buf[4096] = {0};               //
    struct sockaddr localAddr = {0};    //本地地址信息结构图，下面有具体的属性赋值
    struct sockaddr stRemoteAddr = {0}; //对方地址信息
    socklen_t socklen = 0;

    /*
    AF_INET表示IP协议的地址家族
    SOCK_STREAM表示流类型
    */
    socketFd = socket(AF_INET, SOCK_STREAM, 0);
    if (0 > socketFd)
    {
        printf("创建socket失败！\n");
        return 0;
    }

    localAddr.sin_family = AF_INET;                /*该属性表示接收本机或其他机器传输*/
    localAddr.sin_port = htons(PORT);              /*端口号*/
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY); /*IP，括号内容表示本机IP*/

    /*绑定socket文件描述符和本地地址ADDR*/
    if (0 > bind(socketFd, &localAddr, sizeof(localAddr)))
    {
        printf("绑定失败！\n");
        return 0;
    }

    //开启监听 ，第二个参数是最大监听数
    if (0 > listen(socketFd, BACKLOG))
    {
        printf("监听失败！\n");
        return 0;
    }

    printf("socketFd: %d\n", socketFd);
    //在这里阻塞知道接收到消息，参数分别是socket句柄，接收到的地址信息以及大小
    new_fd = accept(socketFd, &stRemoteAddr, &socklen);
    if (0 > new_fd)
    {
        printf("接收失败！\n");
        return 0;
    }
    else
    {
        printf("接收成功！\n");
        //发送内容，参数分别是连接句柄，内容，大小，其他信息（设为0即可）
        send(new_fd, "这是服务器接收成功后发回的信息!", sizeof("这是服务器接收成功后发回的信息!"), 0);
    }

    printf("new_fd: %d\n", new_fd);
    iRecvLen = recv(new_fd, buf, sizeof(buf), 0);
    if (0 >= iRecvLen) //对端关闭连接 返回0
    {
        printf("接收失败或者对端关闭连接！\n");
    }
    else
    {
        printf("buf: %s\n", buf);
    }

    close(new_fd);
    close(socketFd);

    return 0;
}

/*
cd cpp;g++ -g socket.cpp -o socket;./socket;cd ..
cd cpp;rm -rf socket;cd ..
*/