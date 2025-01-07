# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : FASTA_CSV.py
# Time       ：2024/10/19 下午4:02
# Author     ：wleej
# version    ：python 3.10
# Description：
"""

import csv

def custom_fasta_to_csv(fasta_file, csv_file):
    with open(fasta_file, 'r') as f_in, open(csv_file, 'w', newline='') as f_out:
        csv_writer = csv.writer(f_out)
        # 写入表头
        csv_writer.writerow(['ID', 'Description', 'Sequence'])

        # 逐行读取fasta文件，按每三行处理
        lines = f_in.readlines()
        for i in range(0, len(lines), 3):
            # 去掉换行符并按顺序读取ID, Description, Sequence
            id_line = lines[i].strip()
            desc_line = lines[i+1].strip()
            if desc_line.startswith(';'):
                desc_line = desc_line[1:].strip()

            seq_line = lines[i+2].strip()
            # 将数据写入csv
            csv_writer.writerow([id_line, desc_line, seq_line])

# 示例使用
fasta_file = './shuju/ci1c00099_si_002/SI_Files_Case_Studies_Datasets_A-D/Dataset_B/LS_B.fasta'
csv_file = 'LS_B.csv'
custom_fasta_to_csv(fasta_file, csv_file)
