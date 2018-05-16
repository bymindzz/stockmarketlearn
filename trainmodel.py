# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import time
import csv
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_temp = []  # 使用1dcnn处理后的feature
stocks = []  # 使用的100支股票代码
mtime = []  # 日期
price = []  # 价格
target = []  # label
data = []  # 原始feature 包括10天的股票收盘价 和 技术因子

# 开始时间用来排除2005年之前的数据
starttime = time.strptime("2005-1-1", "%Y-%m-%d")
starttimestamp = time.mktime(starttime)


def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def avg_pool_1x2(x):
    # 池化卷积结果（conv2d）池化层采用kernel大小为2*2，步数也为2，周围补0，取最大值。数据量缩小了4倍
    return tf.nn.avg_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


def n_days_BIAS(i, n):
    # n日乖离率
    global price
    five_days_price = 0
    for m in range(n):
        five_days_price += price[i - m]
    price_avg = five_days_price / n
    return (price[i] - price_avg) / price_avg


def n_days_AMP(i, n):
    # n日振幅
    global price
    n_days_price = []
    for m in range(n):
        n_days_price.append(price[i - m])
    return (max(n_days_price) - min(n_days_price)) / min(n_days_price)


def n_days_ROC(i, n):
    # n日涨幅
    global price
    return price[i - n] - price[i] / price[i - n]


def calc_Target(i):
    # 计算5日是涨是跌
    result = price[i + 14] - price[i + 10]
    if result >= 0:
        return 1
    else:
        return 0


def open_Csv(num):
    # 打开csv文件
    print(num)
    return open(r'Stk_DAY_FQ_WithHS_20180310/SH' + num + '.csv', 'r')


def data_Processing(csv_file, stock):
    # 从文件中读取数据并进行初步处理
    global price
    global mtime
    global data
    global target
    global starttimestamp
    global data_temp
    count = 0
    for imfo in csv_file:
        if(count == 0):
            count += 1
            continue
        # day = imfo[1].split('-')
        # dt = datetime.datetime(int(day[0]),int(day[1]),int(day[2]))
        # 把时间转化为时间戳方便计算
        timeArray = time.strptime(imfo.split(',')[1], "%Y-%m-%d")
        timestamp = time.mktime(timeArray)

        if(timestamp >= starttimestamp):
            mtime.append(timestamp)
            price.append(float(imfo.split(',')[5]))
            count += 1

    # 建立一个tensorflow的会话
    sess = tf.InteractiveSession()

    # 官方没有给1dcnn的事例，我就用的2dcnn把高度设为1，这样导致出现问题？
    # 给x，y留出占位符，以便未来填充数据
    xs = tf.placeholder(tf.float32, [None, 16])
    ys = tf.placeholder(tf.float32, [None, 2])
    x_image = tf.reshape(xs, [-1, 1, 16, 1])

    # 设置输入层的W和b
    w = tf.Variable(tf.zeros([16, 2]))
    b = tf.Variable(tf.zeros([2]))

    # 计算输出，采用的函数是softmax（输入的时候是one hot编码）
    y = tf.nn.softmax(tf.matmul(xs, w) + b)

    # 第一个卷积层，1x4的卷积核，输出向量是32维
    w_conv1 = weight_variable([1, 4, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = avg_pool_1x2(h_conv1)

    # 第二层卷积层，输入向量是32维，输出64维，还是1x4的卷积核
    w_conv2 = weight_variable([1, 4, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = avg_pool_1x2(h_conv2)

    # 全连接层的w和b
    w_fc1 = weight_variable([256, 256])
    b_fc1 = bias_variable([256])
    # 此时输出的维数是256维
    h_pool2_flat = tf.reshape(h_pool2, [-1, 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # 设置dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([256, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
    cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv))
    # 设置误差代价以交叉熵的形式
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess.run(tf.global_variables_initializer())

    for i in range(count):
        # data中添加feature

        if i < count - 20:
            data.append([])

            # 5日的技术因子平均值
            f_BIAS = 0
            f_AMP = 0
            f_ROC = 0

            # 10日的技术因子平均值
            t_BIAS = 0
            t_AMP = 0
            t_ROC = 0

            # 20日的技术因子平均值
            # tw_BIAS = 0
            # tw_AMP = 0
            # tw_ROC = 0

            # 30日的技术因子平均值
            # th_BIAS = 0
            # th_AMP = 0
            # th_ROC = 0
            for n in range(10):
                # 添加10天的价格
                data[i].append(price[i + n])
                if i + n >= 5:
                    f_BIAS += n_days_BIAS(i + n, 5)
                    f_AMP += n_days_AMP(i + n, 5)
                    f_ROC += n_days_ROC(i + n, 5)
                if i + n >= 10:
                    t_BIAS += n_days_BIAS(i + n, 10)
                    t_AMP += n_days_AMP(i + n, 10)
                    t_ROC += n_days_ROC(i + n, 10)
                # if i+n >= 20:
                #     tw_BIAS += n_days_BIAS(i + n, 20)
                #     tw_AMP += n_days_AMP(i + n, 20)
                #     tw_ROC += n_days_ROC(i + n, 20)
                # if i+n >= 30:
                #     th_BIAS += n_days_BIAS(i + n, 30)
                #     th_AMP += n_days_AMP(i + n, 30)
                #     th_ROC += n_days_ROC(i + n, 30)

            # 在feature中添加技术因子
            data[i].append(f_BIAS / 10)
            data[i].append(f_AMP / 10)
            data[i].append(f_ROC / 10)
            data[i].append(t_BIAS / 10)
            data[i].append(t_AMP / 10)
            data[i].append(t_ROC / 10)
            # data[i].append(tw_BIAS/10)
            # data[i].append(tw_AMP/10)
            # data[i].append(tw_ROC/10)
            # data[i].append(th_BIAS/10)
            # data[i].append(th_AMP/10)
            # data[i].append(th_ROC/10)

    # 正则化
    sc = StandardScaler()
    sc.fit(data)
    data_std = sc.transform(data)
    # 把正则的模型存到models中
    joblib.dump(sc, "models/sc" + stock + '.model')

    # 把cnn全链接提取出来作为新的特征
    data_temp = sess.run(h_fc1, feed_dict={xs: data_std})

    for i in range(len(data)):
        # label
        target.append(calc_Target(i))


def model(stock):
    # 使用logistic进行训练
    global data
    global target
    global data_temp

    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(
        data_temp, target, test_size=0.1, random_state=0)

    # 训练数据
    lr = LogisticRegressionCV()
    lr.fit(x_train, y_train)

    # 存储模型
    joblib.dump(lr, "models/lr" + stock + '.model')


def main():
    global mtime
    global price
    global target
    global data
    file = open('300.txt')

    for i in file:
        stocks.append(i.strip())

    for stock in stocks:
        csv_file = open_Csv(stock)
        data_Processing(csv_file, stock)
        model(stock)
        mtime = []
        price = []
        target = []
        data = []

if __name__ == '__main__':
    main()
