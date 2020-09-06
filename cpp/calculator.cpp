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
#include <stdio.h>
#include <stdlib.h>
#include <stack>

#define MAX_LENGTH 200

static std::stack<char> symbol;
static std::stack<char> postfix;
static std::stack<double> number;

double cal(double l, double r, char op)
{
    switch (op)
    {
    case '+':
        return l + r;
    case '-':
        return l - r;
    case '*':
        return l * r;
    case '/':
        return l / r;
    }
}

int getPrioripostfix(char c)
{
    if (c == '+' || c == '-')
    {
        return 0;
    }
    else if (c == '*' || c == '/')
    {
        return 1;
    }
    else if (c == '(')
    {
        return -1;
    }
}

void infixTosymbol(char *infix, int charNb)
{
    for (int i = 0; i < charNb; i++)
    {
        if (infix[i] >= 48 && infix[i] <= 57)
        {
            char *start = infix + i;
            while ((infix[i] >= 48 && infix[i] <= 57) || infix[i] == '.')
            {
                i++;
            }
            char *end = infix + i;
            char temp = end[0];
            end[0] = '\0';
            number.push(atof(start));
            end[0] = temp;
        }
        else if (infix[i] == '(')
        {
            symbol.push(infix[i]);
        }
        else if (infix[i] == ')')
        {
            while (symbol.top() != '(')
            {
                postfix.push(symbol.top());
                symbol.pop();
            }
            symbol.pop();
        }
        else
        {
            if (symbol.empty() == true)
            {
                symbol.push(infix[i]);
            }
            else
            {
                int curr = getPrioripostfix(infix[i]);
                int top = getPrioripostfix(symbol.top());
                if (curr > top)
                {
                    symbol.push(infix[i]);
                }
                else
                {
                    while (1)
                    {
                        top = getPrioripostfix(symbol.top());
                        if (symbol.empty() == true || curr > top)
                        {
                            break;
                        }
                        postfix.push(symbol.top());
                        symbol.pop();
                    }
                }
            }
        }
    }
    while (symbol.empty() == false)
    {
        postfix.push(symbol.top());
        symbol.pop();
    }
}

double caculator()
{
    double result = 0;
    char infix[MAX_LENGTH];
    int charNb = 0;
    char c;
    while (1)
    {
        scanf("%c", &c);
        if (c == '\n')
        {
            infix[charNb] = '\0';
            break;
        }
        infix[charNb++] = c;
    }
    infixTosymbol(infix, charNb);

    while (number.empty() == false)
    {
        printf("%lf ", number.top());
        number.pop();
    }
    printf("\n");

    while (postfix.empty() == false)
    {
        printf("%c ", postfix.top());
        postfix.pop();
    }
    printf("\n");

    return result;
}

int main()
{
    caculator();
    return 0;
}
/*
cd cpp;g++ -g calculator.cpp -o calculator;./calculator;cd ..
cd cpp;rm -rf calculator;cd ..
*/