#include <boost/thread.hpp>
#include <stdio.h>

void hello()
{
    printf("Hello world, I'm a thread!\n");
}

int main()
{
    boost::thread thrd(&hello);
    thrd.join();
    return 0;
}
/*
g++ -g boost.cpp -o boost -lpthread -lboost_thread;./boost
rm -rf boost
*/