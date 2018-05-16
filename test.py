#!/usr/bin/env python 
#! -*- coding:utf-8 -*-

import csv

csv_file = open(
    r'C:/Users/72934/Desktop/学校/毕设/股票数据/Stk_DAY_FQ_WithHS_20180310/SH'+'600000'+'.csv', 'r')

data = []
for imfo in csv_file:
    print(imfo.split(",")[1])
    print(imfo.split(",")[5])
print(csv_file)