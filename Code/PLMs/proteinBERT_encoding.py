# pip install protein_bert

from proteinbert import load_pretrained_model
import numpy as np
import pickle
import json
import pandas as pd

"""load the pretrained model"""
pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = "../PLMS" , local_model_dump_file_name = 'epoch_92400_sample_23500000.pkl')

local_representations = []

CSVDATA = open('../data/yourdata.csv',"r",encoding="utf-8")
ls = []
for line in CSVDATA:
    line = line.replace("\n", "")
    ls.append(line.split(","))
se_data = open("../data/yourdata.json","w",encoding='utf-8')
for i in range(1, len(ls)):
    ls[i] = dict(zip(ls[0], ls[i]))
b = json.dumps(ls[1:], sort_keys=True, indent=2, ensure_ascii=False)
se_data.write(b)
se_data.close()

with open('../data/yourdata.json', 'r') as infile :
   your_data = json.load(infile)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def save_array(array, filename):
    with open(filename, 'wb') as file:
        pickle.dump(array, file)

sequences = []
max_len = 0
for i , data in enumerate(your_data) :
    sequences.append(str(data['seq'])) #Sequence
    max_len = max(max_len , len(data['seq']))

max_len += 2
model = pretrained_model_generator.create_model(max_len)
step = 256

"""use the pretrained model to get the local representations of the proteins"""
for i in range(0, len(sequences), step):
    print(i, '/' , len(sequences))
    sequences_ = sequences[i:i+step]
    input_ids = input_encoder.encode_X(sequences_, max_len)
    local_representations_, _ = model.predict(input_ids, batch_size=16)
    if len(local_representations) != 0:
        local_representations = np.concatenate((local_representations, local_representations_), axis=0)
    else:
        local_representations = local_representations_

save_array(local_representations, '../data/data_encoding.pickle')
print('saved successfully!')