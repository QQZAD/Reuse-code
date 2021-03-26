/*
中缀表达式  运算符在两个数字之间    需要考虑运算符优先级
后缀表达式  运算符在两个数字后面    不需要考虑运算符优先级
后缀表达式又称为逆波兰式，没有括号

1.中缀表达式->后缀表达式

a.从左往右遍历，遇到操作数直接输出。
b.遇到左括号直接入栈，入栈后优先级降到最低，确保运算符正常入栈。
c.遇到右括号不断弹出并输出栈顶运算符直到遇到左括号，左括号弹出但不输出。
d.遇到运算符，将该运算符与栈顶运算符进行比较，
如果优先级高于栈顶运算符则直接入栈，
如果优先级低于或等于栈顶运算符则将栈顶运算符弹出并输出，然后比较新的栈顶运算符，
直到优先级高于栈顶运算符或者栈空，再将该运算符入栈。
e.遇到结束符，则弹出并输出栈中所有运算符。

2.计算后缀表达式

a.从左往右遍历，遇到操作数直接入栈。
b.遇到运算符，弹出栈顶的两个操作数，先弹出的在右边后弹出的在左边，计算后将结果入栈。
c.遇到结束符，栈顶保存最终结果。
*/

#include <string>
#include <stack>
using namespace std;

#define MAX_SIZE 200
char deli = ' ';

int cal(int l, int r, char op)
{
    switch (op)
    {
    case '+':
        return l + r;
    case '-':
        return l - r;
    }
    return l * r;
}

int priority(char c)
{
    if (c == '*')
    {
        return 2;
    }
    else if (c == '+' || c == '-')
    {
        return 1;
    }
    return 0;
}

int infixToPostfix(string infix, string &postfix, int len)
{
    stack<char> symbol;
    int p = 0, topChar;
    for (int i = 0; i < len; i++)
    {
        if (infix[i] >= '0' && infix[i] <= '9')
        {
            postfix[p++] = infix[i];
            if (i == len - 1 || (i + 1 < len && (infix[i + 1] < '0' || infix[i + 1] > '9')))
            {
                postfix[p++] = deli;
            }
        }
        else if (infix[i] == '(')
        {
            symbol.push(infix[i]);
        }
        else if (infix[i] == ')')
        {
            while (!symbol.empty())
            {
                topChar = symbol.top();
                symbol.pop();
                if (topChar == '(')
                {
                    break;
                }
                postfix[p++] = topChar;
                postfix[p++] = deli;
            }
        }
        else
        {
            if (!symbol.empty())
            {
                int curr = priority(infix[i]);
                int top = priority(symbol.top());
                if (curr <= top)
                {
                    while (!symbol.empty())
                    {
                        topChar = symbol.top();
                        top = priority(topChar);
                        if (curr > top)
                        {
                            break;
                        }
                        symbol.pop();
                        postfix[p++] = topChar;
                        postfix[p++] = deli;
                    }
                }
            }
            symbol.push(infix[i]);
        }
    }
    while (!symbol.empty())
    {
        postfix[p++] = symbol.top();
        postfix[p++] = deli;
        symbol.pop();
    }
    return p;
}

int solve(string s)
{
    int temp = 0, len = s.size();
    string postfix(MAX_SIZE, 0);
    len = infixToPostfix(s, postfix, len);
    printf("得到后缀表达式：\n%s\n", postfix.c_str());
    stack<int> number;
    for (int i = 0; i < len; i++)
    {
        if (postfix[i] >= '0' && postfix[i] <= '9')
        {
            temp = 10 * temp + postfix[i] - '0';
            if (i + 1 < len && postfix[i + 1] == deli)
            {
                number.push(temp);
                temp = 0;
            }
        }
        else if (postfix[i] != deli)
        {
            int r = number.top();
            number.pop();
            int l = number.top();
            number.pop();
            number.push(cal(l, r, postfix[i]));
        }
    }
    printf("计算结果：%d\n", number.top());
    return number.top();
}

int main()
{
    solve("2-3+4*4(5-8+7)*2");
    return 0;
}
/*
cd cpp;g++ -g calculator.cpp -o calculator;./calculator;cd ..
cd cpp;rm -rf calculator;cd ..
*/