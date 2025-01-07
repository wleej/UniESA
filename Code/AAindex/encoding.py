# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : encoding.py
# Time       ：2024/10/19
# Author     ：wleej
# version    ：python 3.10
# Description：
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

wb  =openpyxl.load_workbook('../data/protein_encoding.xlsx')
sheet=wb['蛋白编码']
#print(wb,sheet)

cells=sheet['B2':'V567']
all_aaindex=[]

for r in cells:
    index_list = []
    for c in r:
        index_list.append(c.value)
    aa_list=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','X']
    aaindex=dict(zip(aa_list,index_list))
    all_aaindex.append(aaindex)

N = 1024

file3=open('../data/aaindex id.csv', 'r')
aaindex_id=[]
line3=file3.readline()
aaindex_id.append(line3.strip())

while line3 !='':
    line3=file3.readline()
    aaindex_id.append(line3.strip())
aaindex_id.pop()
print(aaindex_id)

data = pd.read_csv('your_data.csv', sep=',')
all_seqs = data['Sequence']


for aaindex,encoding_name in zip(all_aaindex,aaindex_id):
    print("current aaindex is: ", aaindex)
    print("current encoding name is: ", encoding_name)
    print("*"*30)
    ls=[]
    for k,v in all_seqs.items():
        numeric_list = []
        seq = v
        print(seq)

        for aa in seq:
            index = aaindex[aa]
            numeric_list.append(index)
        average = sum(numeric_list)/len(numeric_list)
        numeric_list[:] = [i - average for i in numeric_list]
        print(len(numeric_list))

        zero_padding_list = [0] * (N-len(numeric_list))
        numeric_list.extend(zero_padding_list)

        x = np.arange(len(numeric_list))
        half_x = x[range(int(N / 2))]
        fft = np.fft.fft(numeric_list)
        abs_fft = np.abs(fft)
        abs_fft = abs_fft/1024
        half_y = abs_fft[range(int(N/2))]
        ls.append(half_y)
        B = np.array(ls)
      # print(B)
        A = pd.DataFrame(B)
    print(A)
    A.to_csv(f'../output_encoding/{encoding_name}_result_of_encoding.csv')