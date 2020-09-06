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
b.遇到运算符，弹出栈顶的两个操作数，先弹出的在右边后弹出的在左边，计算后将结果压栈。
c.遇到结束符，栈顶保存最终结果。
*/