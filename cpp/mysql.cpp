#include <iostream>
#include <mysql/mysql.h>
#include <string>
using namespace std;

int main()
{
    MYSQL mysql;
    mysql_init(&mysql);
    MYSQL_ROW row;
    MYSQL_RES *rs = nullptr;

    if (!mysql_real_connect(&mysql, "localhost", "root", "12345678", "database_test", 0, nullptr, 0))
    {
        cout << mysql_error(&mysql) << endl;
    }

    string sql = "select * from table_test";
    mysql_query(&mysql, sql.c_str());
    rs = mysql_use_result(&mysql);

    while (row = mysql_fetch_row(rs))
    {
        cout << row[0] << " " << row[1] << endl;
    }

    mysql_free_result(rs);
    mysql_close(&mysql);
}
/*
cd cpp;g++ -g -std=c++17 mysql.cpp `mysql_config --cflags --libs` -o mysql;sudo ./mysql;cd ..
cd cpp;rm -rf mysql;cd ..
*/