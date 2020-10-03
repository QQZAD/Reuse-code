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


def get_x_y_data(data, x_nb, var_orient='vertical'):
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('get_x_y_data var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        data = list(map(list, zip(*data)))
    x_data = list(data[0:x_nb][0])
    y_data = list(data[x_nb:])
    return x_data, y_data
