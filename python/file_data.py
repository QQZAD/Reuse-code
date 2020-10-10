#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import csv
import json
import shutil
import numpy as np
from string import digits


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
        for i in range(len(data)):
            for j in range(len(data[0])):
                f.write(str(data[i][j]))
                if j == len(data[0])-1:
                    f.write('\n')
                else:
                    f.write(' ')
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


def cal_avera_data(dirname, var_orient='vertical'):
    files = os.listdir(dirname)
    temp = read_data(dirname+files[0])
    height = len(temp)
    width = len(temp[0])
    data = [[0 for j in range(width)] for i in range(height)]
    index = 0
    for _ in files:
        _ = dirname+_
        print(_)
        temp = read_data(_)
        index += 1
        for i in range(height):
            for j in range(width):
                data[i][j] += temp[i][j]
    for i in range(height):
        for j in range(width):
            data[i][j] /= index
            if j == 0:
                data[i][j] = int(data[i][j])
    filename = files[0]
    remove_digits = str.maketrans('', '', digits)
    filename = filename.translate(remove_digits)
    print(dirname+filename)
    write_data(dirname+filename, data)


def sort_data_with_x(dirname, var_orient='vertical'):
    files = os.listdir(dirname)
    temp = read_data(dirname+files[0])
    height = len(temp)
    width = len(temp[0])
    data = [[0 for j in range(width)] for i in range(height)]
    for _ in files:
        _ = dirname+_
        print(_)
        temp = read_data(_)
        temp = np.array(temp)
        idex = np.lexsort([temp[:, 0]])
        temp = temp[idex, :]
        for i in range(height):
            for j in range(width):
                if j == 0:
                    data[i][j] = int(temp[i][j])
                else:
                    data[i][j] = temp[i][j]
        # print(data)
        write_data(_, data)


def data_classify():
    if not os.path.exists('ptm_data/'):
        os.mkdir('ptm_data/')
    if not os.path.exists('ptm_data/ids/'):
        os.mkdir('ptm_data/ids/')
    if not os.path.exists('ptm_data/ipsec/'):
        os.mkdir('ptm_data/ipsec/')
    if not os.path.exists('ptm_data/ipv4_router/'):
        os.mkdir('ptm_data/ipv4_router/')
    if not os.path.exists('packet_addr_map'):
        print('没有找到packet_addr_map文件夹')
        exit(1)
    if not os.path.exists('packet_data_copy'):
        print('没有找到packet_data_copy文件夹')
        exit(1)
    files_dc = os.listdir('packet_data_copy')
    files_am = os.listdir('packet_addr_map')
    for _ in files_dc:
        print(_)
        if 'ids' in _:
            shutil.move('packet_data_copy/'+_, 'ptm_data/ids/'+_)
        elif 'ipsec' in _:
            shutil.move('packet_data_copy/'+_, 'ptm_data/ipsec/'+_)
        elif 'ipv4_router' in _:
            shutil.move('packet_data_copy/'+_, 'ptm_data/ipv4_router/'+_)
    for _ in files_am:
        print(_)
        if 'ids' in _:
            shutil.move('packet_addr_map/'+_, 'ptm_data/ids/'+_)
        elif 'ipsec' in _:
            shutil.move('packet_addr_map/'+_, 'ptm_data/ipsec/'+_)
        elif 'ipv4_router' in _:
            shutil.move('packet_addr_map/'+_, 'ptm_data/ipv4_router/'+_)
    os.rmdir('packet_data_copy/')
    os.rmdir('packet_addr_map/')
