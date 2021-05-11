#include <iostream>
#include <memory>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static __attribute((constructor)) void before()
{
    printf("before main\n");
}

static void foo(int n)
{
    printf("foo(int n)");
}

static void foo(char *s)
{
    printf("foo(char* s)");
}

static void constCast()
{
    int *p = nullptr;
    int i = 0;
    const int consti = 24;
    //const_cast用于将const类型指针转换为非const指针
    // p = &consti;
    p = const_cast<int *>(&consti);
}

class Father
{
    virtual void fun()
    {
    }
};

class Child : public Father
{
};

static void staticCast()
{
    // 编译时类型检查
    // 用于各种隐式转换
    // 用于类层次间的上行转换
    Father *pf = new Father();
    Child *pc = new Child();
    pf = static_cast<Father *>(pc);
}

static void dynamicCast()
{
    // 运行时类型检查
    // 用于类层次间的上行转换和下行转换
    // 用于类之间的交叉转换
    Father *pf = new Father();
    Child *pc = new Child();
    // Father类中必须包含虚函数
    pc = dynamic_cast<Child *>(pf);
}

static void reinterpretCast()
{
    // 强制类型转换
    int a = 98;
    int *p = &a;
    char *q = reinterpret_cast<char *>(p);
}

static void smartPointer()
{
    // 智能指针主要用于管理在堆上分配的内存
    // 它将普通的指针封装为一个栈对象
    // 当栈对象的生存周期结束后
    // 会在析构函数中释放掉申请的内存
    // 从而防止内存泄漏

    // unique_ptr
    // 独占式拥有或严格拥有概念
    // 保证同一时间内只有一个智能指针可以指向该对象
    std::unique_ptr<std::string> up1(new std::string("hello world"));
    std::unique_ptr<std::string> up2;
    // up2 = up1;
    up2 = move(up1);
    std::unique_ptr<std::string> up3;
    up3 = std::unique_ptr<std::string>(new std::string("You"));

    //shared_ptr
    // 实现共享式拥有概念
    // 多个智能指针可以指向相同对象
    // 该对象和其相关资源会在最后一个引用被销毁时候释放
    // 它使用计数机制来表明资源被几个指针共享
    int *sp = new int(30);
    std::shared_ptr<int> sp1(sp);
    std::shared_ptr<int> sp2 = std::make_shared<int>(20);
    std::shared_ptr<int> sp3(sp2);
    printf("sp1.use_count() = %ld value = %d\n", sp1.use_count(), *sp1);
    printf("sp2.use_count() = %ld value = %d\n", sp2.use_count(), *sp2);
    printf("sp3.use_count() = %ld value = %d\n", sp3.use_count(), *sp3);

    // weak_ptr
    // 弱指针，旨在解决内存泄漏问题
    // 指向一个shared_ptr管理的对象
    // weak_ptr只是提供了对管理对象的一个访问手段
    // weak_ptr是用来解决shared_ptr相互引用时的死锁问题
    // 如果说两个shared_ptr相互引用
    // 那么这两个指针的引用计数永远不可能下降为0
    // 资源永远不会释放
    std::shared_ptr<std::string> wp1 = std::make_shared<std::string>("hello world");
    // wp2作为旁观者管理wp1的资源使用情况
    // std::weak_ptr<std::string> wp2 = wp1;
    std::weak_ptr<std::string> wp2(wp1);
    // 获取所管理的对象的强引用(shared_ptr)
    // 不能通过weak_ptr直接访问对象的方法
    std::shared_ptr<std::string> wp3 = wp2.lock();
    if (wp3 != nullptr)
    {
        std::cout << *wp3 << std::endl;
    }
    std::cout << wp2.expired() << std::endl;
    wp2.reset();
    wp3 = wp2.lock();
    if (wp3 == nullptr)
    {
        std::cout << "nullptr" << std::endl;
    }
    std::cout << wp2.expired() << std::endl;
}

//extern "C"指示按C风格编译代码
extern "C"
{
    static void useForkWaitExec()
    {
        int status = 0;
        int a = 0;
        a++;
        pid_t result = fork();
        if (result != 0)
        {
            printf("我是父进程 a=%d 子进程的pid为%d\n", a, result);
            wait(&status);
            printf("父进程结束\n");
        }
        else
        {
            printf("我是子进程 a=%d\n", a);
            char *p[] = {NULL};
            execv("./link", p);
            sleep(3);
            printf("子进程结束\n");
        }
    }

    static void useSelect()
    {
        int a = 0;
        a++;
        pid_t result = fork();
        if (result != 0)
        {
            printf("我是父进程 a=%d 子进程的pid为%d\n", a, result);
        }
        else
        {
            printf("我是子进程 a=%d\n", a);
        }
    }
}

#include <vector>
#include <list>
#include <deque>
#include <set>
#include <map>
#include <unordered_map>

// STL
// 容器、算法、迭代器、函数对象（仿函数）、适配器、内存分配器
// 内存分配器给容器分配存储空间
// 算法通过迭代器获取容器中的内容
// 函数对象协助算法完成各种操作
// 适配器用来套接适配函数对象

// 内存的配置 alloc::allocate()
// 对象的构造 ::construct()
// 对象的析构 ::destroy()
// 内存的释放 alloc::deallocate()

// Iterator模式是运用于聚合对象的一种模式
// 把不同集合类的访问逻辑抽象出来
// 使得不用暴露集合内部的结构而达到循环遍历集合的效果

// 迭代器本质封装了原生指针
// 相当于一种智能指针
// 可以根据不同类型的数据结构来实现不同的操作
// 迭代器返回的是对象引用而不是对象的值

static void useVector()
{
    // 底层实现了动态数组
    // 连续存储结构
    // 经常随机访问
    // 不经常对非尾节点进行插入删除
    // 两倍容量增长
    std::vector<int> value;
    for (int i = 0; i < 26; i++)
    {
        // value.insert(value.begin() + i, i + 1);
        value.push_back(i + 1);
    }
    // std::vector<int> value(25, -1);
    std::vector<int>::iterator it;
    for (it = value.begin(); it != value.end(); it++)
    {
        if (*it == 5)
        {
            value.erase(it);
        }
        std::cout << *it << std::endl;
    }
    std::cout << "size:" << value.size() << std::endl;
    std::cout << "capacity:" << value.capacity() << std::endl;
    // 改变当前容器的元素数量
    value.resize(2 * value.size());
    // 改变当前容器的最大容量
    value.reserve(2 * value.capacity());
    for (it = value.begin(); it != value.end(); it++)
    {
        std::cout << *it << std::endl;
    }
    std::cout << "size:" << value.size() << std::endl;
    std::cout << "capacity:" << value.capacity() << std::endl;
}

static void useDeque()
{
    // 双端队列
    // 连续存储结构
    // 为了随机访问用数组实现
    // 为了在双端扩容用数组存不同连续空间的指针
    // 在中间部分安插元素则比较费时
    // 因为必须移动其它元素
    std::deque<int> value;
    for (int i = 0; i < 13; i++)
    {
        value.push_back(i + 1);
    }
    for (int i = 0; i < 13; i++)
    {
        value.push_front(i + 1);
    }
    std::deque<int>::iterator it;
    for (it = value.begin(); it != value.end(); it++)
    {
        if (*it == 5)
        {
            value.erase(it);
        }
        std::cout << *it << std::endl;
    }
}

static void useList()
{
    // 底层实现了双向链表
    // 非连续存储结构
    // 快速的插入和删除
    // 随机访问却比较慢
    std::list<int> value;
    for (int i = 0; i < 13; i++)
    {
        value.push_back(i + 1);
    }
    for (int i = 0; i < 13; i++)
    {
        value.push_front(i + 1);
    }
    std::list<int>::iterator it;
    for (it = value.begin(); it != value.end(); it++)
    {
        if (*it == 5)
        {
            // value.erase(it);
            it = value.erase(it);
        }
        std::cout << *it << std::endl;
    }
}

static void useSet()
{
    std::set<char> value;
    for (int i = 26; i >= 0; i--)
    {
        value.insert(i + 'a' - 1);
    }
    value.erase(value.begin());
    for (int i = 0; i < 5; i++)
    {
        value.insert(i + 'a');
    }
    std::set<char>::iterator it;
    // set自动排序，底层实现了RBT
    for (it = value.begin(); it != value.end(); it++)
    {
        // 迭代器指向的元素是只读的
        // *it = '_';
        std::cout << *it << std::endl;
    }
    // set不支持[]操作符
    // value[0] = '0';
}

static void useMap()
{
    std::map<int, char> keyValue;
    for (int i = 26; i >= 0; i--)
    {
        keyValue.insert(std::pair<int, char>(i, i + 'a' - 1));
    }
    keyValue.erase(keyValue.begin());
    std::map<int, char>::iterator it, temp;
    // map自动排序，底层实现了RBT
    for (it = keyValue.begin(); it != keyValue.end(); it++)
    {
        // 迭代器指向的关键字是只读的
        // it->first = 1;
        // 迭代器指向的值是可写的
        // it->second = '_';
        if (it->second == 'd')
        {
            // keyValue.erase(it);
            keyValue.erase(it++);
        }
        std::cout << it->first << "-" << it->second << std::endl;
    }
    keyValue[0] = '0';
    std::map<int, char>::iterator itf;
    itf = keyValue.find(9);
    if (itf != keyValue.end())
    {
        std::cout << itf->second << std::endl;
    }
}

static void useMultimap()
{
    std::multimap<int, char> keyValue;
    for (int i = 26; i >= 0; i--)
    {
        keyValue.insert(std::pair<int, char>(i, i + 'a' - 1));
    }
    keyValue.insert(std::pair<int, char>(26, '.'));
    std::multimap<int, char>::iterator it;
    for (it = keyValue.begin(); it != keyValue.end(); it++)
    {
        std::cout << it->first << "-" << it->second << std::endl;
    }
}

static void useUnorderedMap()
{
    std::unordered_map<int, char> keyValue;
    for (int i = 26; i >= 0; i--)
    {
        keyValue.insert(std::pair<int, char>(i, i + 'a' - 1));
    }
    keyValue.erase(keyValue.begin());
    std::unordered_map<int, char>::iterator it;
    // unordered_map是无序的，底层实现了hash表
    for (it = keyValue.begin(); it != keyValue.end(); it++)
    {
        // 迭代器指向的关键字是只读的
        // it->first = 1;
        // 迭代器指向的值是可写的
        // it->second = '_';
        std::cout << it->first << "-" << it->second << std::endl;
    }
    keyValue[0] = '0';
    std::unordered_map<int, char>::iterator itf;
    itf = keyValue.find(9);
    if (itf != keyValue.end())
    {
        std::cout << itf->second << std::endl;
    }
}

static void useUnorderedMultiMap()
{
    std::unordered_multimap<int, char> keyValue;
    for (int i = 26; i >= 0; i--)
    {
        keyValue.insert(std::pair<int, char>(i, i + 'a' - 1));
    }
    keyValue.insert(std::pair<int, char>(26, '.'));
    std::unordered_multimap<int, char>::iterator it;
    for (it = keyValue.begin(); it != keyValue.end(); it++)
    {
        std::cout << it->first << "-" << it->second << std::endl;
    }
}

#include <stack>

// 返回每个数后面第一个比它大的数
// 若没有则结果为-1
static std::vector<int> findMax(std::vector<int> num)
{
    std::vector<int> res(num.size(), -1);
    int i = 0;
    std::stack<int> s;
    while (i < num.size())
    {
        if (s.empty() || num[s.top()] >= num[i])
        {
            s.push(i++);
        }
        else
        {
            res[s.top()] = num[i];
            s.pop();
        }
    }
    for (int i = 0; i < res.size(); i++)
    {
        std::cout << res[i] << std::endl;
    }
    return res;
}

int main()
{
    printf("main\n");
    // useForkWaitExec();
    // smartPointer();
    // useVector();
    // useDeque();
    // useList();
    // useSet();
    // useMap();
    // useMultimap();
    // useUnorderedMap();
    // useUnorderedMultiMap();
    std::vector<int> t;
    t.push_back(2);
    t.push_back(5);
    t.push_back(4);
    t.push_back(7);
    t.push_back(6);
    findMax(t);
    return 0;
}

/*
c++17
c++14
c++11
c++03
c++98
cd cpp;g++ -g -std=c++17 cpp.cpp -o cpp;./cpp;cd ..
cd cpp;rm -rf cpp;cd ..
*/