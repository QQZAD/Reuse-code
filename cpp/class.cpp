#include <stdio.h>

class Const
{
public:
    int val;
    Const() {}
    Const(int val)
    {
        this->val = val;
    }
    // 拷贝构造函数必须传递引用
    Const(const Const &c)
    {
        this->val = c.val;
    }
    // 拷贝构造函数不能值传递
    // 如果参数不是引用，则永远不会调用成功
    // 为了调用拷贝构造函数，我们必须拷贝它的实参
    // 但为了拷贝实参，我们又必须调用拷贝构造函数
    // Const(const Const c)
    // {
    //     this->val = c.val;
    // }
    // Const &operator=(const Const &c)
    // {
    //     this->val = c.val;
    //     return *this;
    // }
    // 拷贝赋值函数可以值传递
    Const &operator=(const Const c)
    {
        this->val = c.val;
        return *this;
    }
    void print(int val) {}
    void print(int &val) {}
    void print() { printf("Const val=%d\n", val); }
    void print() const {}
};

// 隐式类型转换
static void invokeConst(Const c)
{
    printf("Const val=%d\n", c.val);
}

class Test
{
public:
    int &ref;
    // !定义引用成员时禁止使用默认构造函数
    // Test() {}
    // Test(int &ref) { this->ref = ref; }
    // !定义引用成员时需要使用形参列表
    // !形参类型必须同样是引用
    Test(int &ref) : ref(ref) {}
    // !使用成员函数也可以达到同样的效果
    void set(int &ref) { this->ref = ref; }
};

// 左值、右值
// 左值引用、右值引用、通用引用
static void valueAndRef()
{
    // a是左值，既能出现在=左边又能出现在=右边的值
    // 可修改，可寻址的变量，持久性
    int a;
    // 3是右值，只能出现在=右边的值
    // 不可修改，不可寻址的常量，短暂性
    a = 3;

    // 左值引用，只能绑定左值
    int &x = a;
    // x*6是一个右值
    // int &x1 = x * 6;

    // 常量左值引用，既可以绑定左值又可以绑定右值
    const int &y = x;
    const int &y1 = x * 6;

    // 右值引用，只能绑定右值
    int &&z = x * 6;
    // x是一个左值
    // int &&z1 = x;

    // 编译器将右值引用z视为左值
    z = 3;
}

int main()
{
    invokeConst(5);
    valueAndRef();
}
/*
cd cpp;g++ -g -std=c++17 class.cpp -o class;./class;cd ..
cd cpp;rm -rf class;cd ..
*/