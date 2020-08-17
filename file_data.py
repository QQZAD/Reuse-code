#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import csv
import json


def read_data(path):
    suffix = os.path.splitext(path)[1]
    f = open(path, 'r')
    if suffix == '.csv':
        data = []
        reader = csv.reader(f)
        for i in reader:
            temp = []
            for j in i:
                temp.append(float(j))
            data.append(temp)
    elif suffix == '.json':
        data = json.load(f)
    else:
        data = f.read()
    f.close()
    return data


def write_data(path, data):
    suffix = os.path.splitext(path)[1]
    f = open(path, 'w')
    if suffix == '.csv':
        if isinstance(data, list) or isinstance(data, tuple):
            writer = csv.writer(f)
            for _ in data:
                writer.writerow(_)
        else:
            print('csv文件用于保存列表类型或元组类型！')
    elif suffix == '.json':
        if isinstance(data, dict):
            json_str = json.dumps(data, indent=4)
            f.write(json_str)
        else:
            print('json文件用于保存字典类型！')
    else:
        f.write(str(data))
    f.close()
