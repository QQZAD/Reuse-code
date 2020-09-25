#include <stdio.h>

#define BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BINARY(byte)               \
    (byte & 0x80 ? '1' : '0'),     \
        (byte & 0x40 ? '1' : '0'), \
        (byte & 0x20 ? '1' : '0'), \
        (byte & 0x10 ? '1' : '0'), \
        (byte & 0x08 ? '1' : '0'), \
        (byte & 0x04 ? '1' : '0'), \
        (byte & 0x02 ? '1' : '0'), \
        (byte & 0x01 ? '1' : '0')

void printBinary(int bytes, void *data)
{
    /*
    *整数在内存中以二进制补码形式存储，方便计算
    *低位地址存储数值的低位

    *float 4bytes 32bits
    Sign 1bit 符号 0表示正数，1表示负数
    Exponent 8bits 指数 无符号整数 底数为2 偏移量为2^(8-1)-1=127 0b 01111111 真值+偏移量=实际存储值
    Mantissa 23bits 尾数 表示小数点后的数

    *double 8bytes 64bits
    Sign 1bit 符号 0表示正数，1表示负数
    Exponent 11bits 指数 无符号整数 底数为2 偏移量为2^(11-1)-1=1023 0b 01111111 111 真值+偏移量=实际存储值
    Mantissa 52bits 尾数 表示小数点后的数
    */
    printf("内存地址由高->低\n");
    printf("0b ");
    char *p = (char *)data;
    for (int i = bytes - 1; i >= 0; i--)
    {
        printf(BINARY_PATTERN, BINARY(p[i]));
        if (i > 0)
        {
            printf(" ");
        }
    }
    printf("\n");
}

void dataType()
{
    char c = 97;
    printf("c=%c\n", c);
    printBinary(sizeof(char), (void *)&c);

    c = -97;
    printf("c=%d\n", c);
    printBinary(sizeof(char), (void *)&c);

    unsigned char uc = 97;
    printf("uc=%u\n", uc);
    printBinary(sizeof(unsigned char), (void *)&uc);

    short s = -16;
    printf("s=%hd\n", s);
    printBinary(sizeof(short), (void *)&s);

    unsigned short us = 16U;
    printf("us=%hu\n", us);
    printBinary(sizeof(unsigned short), (void *)&us);

    int i = -16;
    printf("i=%d\n", i);
    printBinary(sizeof(int), (void *)&i);

    unsigned int ui = 16U;
    printf("ui=%u\n", ui);
    printBinary(sizeof(unsigned int), (void *)&ui);

    long l = -16L;
    printf("l=%ld\n", l);
    printBinary(sizeof(long), (void *)&l);

    unsigned long ul = 16UL;
    printf("ul=%lu\n", ul);
    printBinary(sizeof(unsigned long), (void *)&ul);

    long long ll = -16LL;
    printf("ll=%lld\n", ll);
    printBinary(sizeof(long long), (void *)&ll);

    unsigned long long ull = 16ULL;
    printf("ull=%llu\n", ull);
    printBinary(sizeof(unsigned long long), (void *)&ull);

    float f = 0.03125F; //1*2^(-5)
    printf("f=%f\n", f);
    printf("f=%e\n", f);
    printBinary(sizeof(float), (void *)&f);
    f = -0.03125F;
    printf("f=%f\n", f);
    printBinary(sizeof(float), (void *)&f);

    double d = 0.03125;
    printf("d=%lf\n", d);
    printBinary(sizeof(double), (void *)&d);
    d = -0.03125;
    printf("d=%lf\n", d);
    printBinary(sizeof(double), (void *)&d);

    long double ld = 0.03125L;
    printf("ld=%Lf\n", ld);
    printBinary(sizeof(long double), (void *)&ld);
    ld = -0.03125L;
    printf("ld=%Lf\n", ld);
    printBinary(sizeof(long double), (void *)&ld);
}

void binary()
{
    int hex = 0x20;
    printf("0x%x\n", hex);
    int oct = 040;
    printf("0%o\n", oct);
    int dec = 32;
    printf("%d\n", dec);
    int bin = 0b100000;
    printBinary(sizeof(int), (void *)&bin);
}

void bitsOperate()
{
    int i = 9;
    printBinary(sizeof(int), (void *)&i);
    i |= 0b10101;
    printBinary(sizeof(int), (void *)&i);
    i &= 0b10101;
    printBinary(sizeof(int), (void *)&i);
    i ^= 0b01010;
    printBinary(sizeof(int), (void *)&i);
    i = ~i;
    printBinary(sizeof(int), (void *)&i);
    /*左移一位代表乘2，右边补0*/
    i <<= 5;
    printBinary(sizeof(int), (void *)&i);
    /*
    右移一位代表除2，
    对于有符号数，左边补符号位
    对于无符号数，左边补0
    */
    i >>= 3;
    printBinary(sizeof(int), (void *)&i);
    i &= 0xFfFfFf00;
    printBinary(sizeof(int), (void *)&i);
    i &= 0x0fFfFfFf;
    printBinary(sizeof(int), (void *)&i);
}

int main()
{
    bitsOperate();
    return 0;
}

/*
cd cpp;g++ -g binary.cpp -o binary;./binary;cd ..
cd cpp;rm -rf binary;cd ..
*/