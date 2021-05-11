#include <stdio.h>

static void readCharSet(int len)
{
    char str[len];
    // 特殊字符[, ], -, ^
    // int res = scanf("%[a-z]", str);
    // int res = scanf("%[z-a]", str);
    // int res = scanf("%[az]", str);
    // int res = scanf("%[za]", str);
    // int res = scanf("%[0-9]", str);
    // int res = scanf("%[9-0]", str);
    // int res = scanf("%[09]", str);
    // int res = scanf("%[90]", str);
    // int res = scanf("%[^.]", str);
    // int res = scanf("%[-*]", str);
    // int res = scanf("%3[^.]", str);
    //读取[和]
    // int res = scanf("%[][]", str);
    //读取-
    // int res = scanf("%[[-]]", str);
    //读取^
    // int res = scanf("%[[^]]", str);
    int res = scanf("%[](){}*+[-][]", str);
    if (res)
    {
        printf("%s\n", str);
    }
}

int main()
{
    readCharSet(64);
}

/*
cd cpp;g++ -g -std=c++17 input.cpp -o input;./input;cd ..
cd cpp;rm -rf input;cd ..
*/