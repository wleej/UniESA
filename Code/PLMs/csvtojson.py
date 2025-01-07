# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : csvtojson.py
# Time       ：2024/10/22 上午9:11
# Author     ：wleej
# version    ：python 3.10
# Description：
"""
import csv
import json

def csv_to_json(csv_file, json_file):
    data = []
    with open(csv_file, mode='r', newline='') as f_in:
        csv_reader = csv.DictReader(f_in)

        for row in csv_reader:
            data.append({
                'sequence': row['Sequence'],
                'label': row['Description']
            })
    with open(json_file, mode='w') as f_out:
        json.dump(data, f_out, indent=4)

csv_file = '../data/yourdata.csv'
json_file = '../data/yourdata.json'
csv_to_json(csv_file, json_file)

