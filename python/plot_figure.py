#!/usr/bin/python3
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import file_data as fd
import numpy as np
import copy


def plot_line(x_data, y_data, save_path, x_label, y_label, var_orient='vertical'):
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        y_data = list(map(list, zip(*y_data)))
    x_number = len(y_data[0])
    y_number = len(y_data)
    if len(x_data) != x_number:
        print('x_data和y_data不匹配！')
        exit(1)
    # 设置y_data的label、color、linestyle、marker
    # label = ['']
    # label = ['Normal copy mode',
    #          'Zero-copy mode']
    # label = ['Copy 1024 packets at a time',
    #          'Copy 2048 packets at a time']
    label = ['Serial PCIe', 'Parallel PCIe', 'Hybrid PCIe']
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white', 'cyan', 'darksalmon', 'gold', 'crimson'
    # color = ['red']
    # color = ['red', 'green']
    color = ['red', 'green', 'black']
    # 线条风格 '-', '--', '-.', ':', 'None'
    # linestyle = ['-']
    # linestyle = ['-', '--']
    linestyle = ['-', '--', '-.']
    # 线条标记 'o', 'd', 'D', 'h', 'H', '_', '8', 'p', ',', '.', 's', '+', '*', 'x', '^', 'v', '<', '>', '|'
    # marker = ['o']
    # marker = ['v', '^']
    marker = ['v', '^', 'o']
    if y_number != len(label) or y_number != len(color) or y_number != len(linestyle) or y_number != len(marker):
        print('y_data的label、color、linestyle、marker没有被正确设置！')
        exit(1)
    # plt.title('title')
    # 设置字体和大小
    plt.rc('font', family='Times New Roman', size=17)
    # 设置x轴的标签
    plt.xlabel(x_label)
    # 设置x轴的范围
    # plt.xlim([0, 5])
    # 设置y轴的标签
    plt.ylabel(y_label)
    # 设置y轴的范围
    # plt.ylim([-5, 25])
    for _ in range(y_number):
        plt.plot(x_data, y_data[_], label=label[_],
                 color=color[_], linestyle=linestyle[_], linewidth=3)
        plt.scatter(x_data, y_data[_], color=color[_], marker=marker[_], s=50)
    # 'x' 'y' 'both'
    plt.grid(axis='y')
    plt.legend()
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(9, 3.5)
    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def plot_bar(x_data, y_data, save_path, x_label, y_label, var_orient='vertical'):
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        y_data = list(map(list, zip(*y_data)))
    x_number = len(y_data[0])
    y_number = len(y_data)
    if len(x_data) != x_number:
        print('x_data和y_data不匹配！')
        exit(1)
    # 设置y_data的label、color、hatch
    label = ['label1', 'label2', 'label3']
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white', 'cyan', 'darksalmon', 'gold', 'crimson'
    color = ['green', 'red', 'blue']
    # 填充效果 '/', '|', '-', '+', 'x', 'o', 'O', '.', '*', ' '
    hatch = ['x', '.', '|']
    if y_number != len(label) or y_number != len(color) or y_number != len(hatch):
        print('y_data的label、color、hatch没有被正确设置！')
        exit(1)
    # plt.title('title')
    # 设置字体和大小
    plt.rc('font', family='Times New Roman', size=17)
    # 设置柱状体的宽度
    bar_width = 0.20
    # 设置x轴的标签
    plt.xlabel(x_label)
    # 设置x轴的范围
    # plt.xlim([0, 5])
    # 设置x轴的刻度
    plt.xticks(np.arange(x_number)+bar_width*(y_number/2-0.5), x_data)
    # 设置y轴的标签
    plt.ylabel(y_label)
    # 设置y轴的范围
    # plt.ylim([-5, 25])
    for _ in range(y_number):
        plt.bar(np.arange(x_number)+_*bar_width,
                y_data[_], label=label[_], color=color[_], hatch=hatch[_], width=bar_width)
    # 'x' 'y' 'both'
    plt.grid(axis='y')
    plt.legend()
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(9, 3.5)
    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    print('若图像出现重叠现象，调小bar_width的值！')
    plt.show()


def plot_stacked_bar(x_data, y_data, save_path, x_label, y_label, var_orient='vertical'):
    print('在堆叠柱状图中序号小的y_data位于柱状图下层')
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        y_data = list(map(list, zip(*y_data)))
    x_number = len(y_data[0])
    y_number = len(y_data)
    if len(x_data) != x_number:
        print('x_data和y_data不匹配！')
        exit(1)
    # 计算堆叠后的_y_data
    _y_data = copy.deepcopy(y_data)
    for i in range(y_number):
        for j in range(x_number):
            if i-1 >= 0:
                _y_data[i][j] += _y_data[i-1][j]
    # 百分比保留几位小数
    decimal = 2
    # 计算堆叠柱状图的百分比
    prop_y_data = copy.deepcopy(y_data)
    for i in range(y_number):
        for j in range(x_number):
            prop_y_data[i][j] = round(
                y_data[i][j]/_y_data[y_number-1][j], decimal)
    # 设置y_data的label、color、hatch
    label = ['label1', 'label2', 'label3']
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white', 'cyan', 'darksalmon', 'gold', 'crimson'
    color = ['green', 'red', 'darksalmon']
    # 填充效果 '/', '|', '-', '+', 'x', 'o', 'O', '.', '*', ' '
    hatch = [' ', ' ', ' ']
    if y_number != len(label) or y_number != len(color) or y_number != len(hatch):
        print('y_data的label、color、hatch没有被正确设置！')
        exit(1)
    # plt.title('title')
    # 设置字体和大小
    plt.rc('font', family='Times New Roman', size=17)
    # 设置柱状体的宽度
    bar_width = 0.6
    # 设置x轴的标签
    plt.xlabel(x_label)
    # 设置x轴的范围
    # plt.xlim([0, 5])
    # 设置x轴的刻度
    plt.xticks(np.arange(x_number), x_data)
    # 设置y轴的标签
    plt.ylabel(y_label)
    # 设置y轴的范围
    plt.ylim([0, 65])
    for _ in range(y_number-1, - 1, - 1):
        plt.bar(np.arange(x_number),
                _y_data[_], label=label[_], color=color[_], hatch=hatch[_], width=bar_width)
    # 添加堆叠柱状图的百分比
    for i in range(y_number):
        for j in range(x_number):
            plt.text(j, _y_data[i][j]-y_data[i][j]/2, s=str(
                prop_y_data[i][j]), ha='center', va='center', fontsize=14)
    # 'x' 'y' 'both'
    plt.grid(axis='y')
    plt.legend()
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(9, 3.5)
    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    print('若图像出现重叠现象，调小bar_width的值！')
    plt.show()


def benchmark():
    x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_data = [[3, 13, 5, 7, 9, 10, 2, 1, 0],
              [6, 3, 8, 4, 9, 14, 9, 2, 7],
              [7, 9, 5, 8, 5, 3, 9, 7, 2]]
    plot_line(x_data, y_data, 'data1.eps', 'xlabel', 'ylabel', 'horizon')
    plot_bar(x_data, y_data, 'data2.eps', 'xlabel', 'ylabel', 'horizon')
    plot_stacked_bar(x_data, y_data, 'data3.eps',
                     'xlabel', 'ylabel', 'horizon')

    # cd python;rm -rf __pycache__;cd ..;rm -rf data1.eps data2.eps data3.eps


if __name__ == '__main__':
    benchmark()
    # x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # y_data = [[12, 13, 52, 61, 23, 45, 15, 44, 42, 39]]
    # plot_line(x_data, y_data, 'data.eps', 'Number of flows',
    #           'Average latency of data copy (s)', 'horizon')

    # data = fd.read_data('copy_time_batch/experiment1.txt')
    # x_data, y_data = fd.get_x_y_data(data)
    # plot_line(x_data, y_data, 'data1.eps', 'Batch size',
    #           'The time overhead (s)', 'horizon')

    # fd.data_classify()
    # fd.cal_avera_data('ptm_data/ids/')
    # fd.cal_avera_data('ptm_data/ipsec/')
    # fd.cal_avera_data('ptm_data/ipv4_router/')
    # fd.cal_avera_data('ptm_data/nat/')

    # x_data, y_data = fd.read_dir_data('ptm_data/ids/')
    # plot_line(x_data, y_data, 'data1.eps', 'Batch size',
    #           'The λ of IDS (s)', 'horizon')

    # x_data, y_data = fd.read_dir_data('ptm_data/ipsec/')
    # plot_line(x_data, y_data, 'data2.eps', 'Batch size',
    #           'The λ of IPsec (s)', 'horizon')

    # x_data, y_data = fd.read_dir_data('ptm_data/ipv4_router/')
    # plot_line(x_data, y_data, 'data3.eps', 'Batch size',
    #           'The λ of IPv4 Router (s)', 'horizon')

    # x_data, y_data = fd.read_dir_data('ptm_data/nat/')
    # plot_line(x_data, y_data, 'data4.eps', 'Batch size',
    #           'The λ of NAT (s)', 'horizon')

    # fd.cal_avera_data('pcie_competition/')
    # x_data, y_data = fd.read_dir_data('pcie_competition/')
    # plot_line(x_data, y_data, 'data5.eps', 'Number of flows',
    #           'Average latency of data copy (s)', 'horizon')

    # cd python;rm -rf __pycache__;cd ..;rm -rf data.eps data.csv data1.eps data2.eps data3.eps data4.eps data5.eps
