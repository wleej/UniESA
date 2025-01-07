# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : fitness_encoding.py
# Time       ：2024/9/2 下午8:25
# Author     ：wleej
# version    ：python 3.10
# Description：
"""
import json
from rdkit import Chem
from collections import defaultdict
import numpy as np
import pickle
import tqdm

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def save_array(array, filename):
    with open(filename, 'wb') as file:
        pickle.dump(array, file)

with open('../data/yourdata.json', 'r') as infile :
    ac_data = json.load(infile)

fitness = []

for data in tqdm.tqdm(ac_data):
    fitness.append(float(data['label']))

save_array(fitness, '../data/fitness.pickle')
print('saved successfully!')