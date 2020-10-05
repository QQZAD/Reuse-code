#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import csv
import json
import numpy as np


def read_data(path):
    suffix = os.path.splitext(path)[1]
    f = open(path, 'r')
    data = []
    if suffix == '.csv':
        reader = csv.reader(f)
        for i in reader:
            temp = []
            for j in i:
                temp.append(float(j))
            data.append(temp)
    elif suffix == '.json':
        data = json.load(f)
    else:
        lines = f.readlines()
        for line in lines:
            temp = line.strip('\n').split(' ')
            for _ in range(len(temp)):
                temp[_] = float(temp[_])
            data.append(temp)
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


def get_x_y_data(data, var_orient='vertical'):
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('get_x_y_data var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        data = list(map(list, zip(*data)))
    x_data = list(data[0:1][0])
    y_data = list(data[1:])
    return x_data, y_data


def read_dir_data(dirname, var_orient='vertical'):
    files = os.listdir(dirname)
    y_data = []
    for _ in files:
        _ = dirname+_
        print(_)
        data = read_data(_)
        # print(data)
        x_data, _y_data = get_x_y_data(data, var_orient)
        y_data.append(_y_data[0])
    print(x_data)
    print(y_data)
    return x_data, y_data
