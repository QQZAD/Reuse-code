#include <boost/version.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/noncopyable.hpp>
#include <iostream>
#include <vector>

typedef boost::mutex::scoped_lock lock;

class BoundedBuffer : private boost::noncopyable
{
private:
    int begin, end, buffered;
    std::vector<int> circularBuf;
    boost::condition bufferNotFull, bufferNotEmpty;
    boost::mutex monitor;

public:
    BoundedBuffer(int n) : begin(0), end(0), buffered(0), circularBuf(n) {}

    void send(int m)
    {
        lock lk(monitor);
        while (buffered == circularBuf.size())
        {
            bufferNotFull.wait(lk);
        }

        circularBuf[end] = m;
        end = (end + 1) % circularBuf.size();
        ++buffered;
        bufferNotEmpty.notify_one();
    }

    int receive()
    {
        lock lk(monitor);
        while (buffered == 0)
        {
            bufferNotEmpty.wait(lk);
        }

        int i = circularBuf[begin];
        begin = (begin + 1) % circularBuf.size();
        --buffered;
        bufferNotFull.notify_one();
        return i;
    }
};
BoundedBuffer buf(2);

static void sender()
{
    int n = 0;
    while (n < 100)
    {
        buf.send(n);
        std::cout << "sent: " << n << std::endl;
        ++n;
    }
    buf.send(-1);
}

static void receiver()
{
    int n;
    do
    {
        n = buf.receive();
        std::cout << "received: " << n << std::endl;
    } while (n != -1); // -1 indicates end of buffer
}

int main()
{
    std::cout << "boost版本号" << std::endl
              << BOOST_LIB_VERSION << std::endl;
    boost::thread thrd1(&sender);
    boost::thread thrd2(&receiver);
    thrd1.join();
    thrd2.join();
    return 0;
}
/*
cd cpp;g++ -g -std=c++17 boost.cpp -o boost -lpthread -lboost_thread;./boost;cd ..
cd cpp;rm -rf boost;cd ..
*/