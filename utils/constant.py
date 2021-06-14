import torch


CUDA = torch.cuda.is_available()
pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
hieve_label_dict={"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
hieve_num_dict = {0: "SuperSub", 1: "SubSuper", 2: "Coref", 3: "NoRel"}

mypath_TB = './datasets/MATRES/TBAQ-cleaned/TimeBank/' # after correction
mypath_AQ = './datasets/MATRES/TBAQ-cleaned/AQUAINT/' 
mypath_PL = './datasets/MATRES/te3-platinum/'
MATRES_timebank = './datasets/MATRES/timebank.txt'
MATRES_aquaint = './datasets/MATRES/aquaint.txt'
MATRES_platinum = './datasets/MATRES/platinum.txt'
temp_label_map = {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}

i2b2_label_dict = {'BEFORE': 0, 'AFTER': 1, 'SIMULTANEOUS': 2, 'OVERLAP': 2, 'simultaneous': 2, 
                    'BEGUN_BY': 1, 'ENDED_BY': 0, 'DURING': 2,'BEFORE_OVERLAP': 0}
tbd_label_dict = { 'BEFORE': 0, 'AFTER': 1, 'INCLUDES': 2, 'IS_INCLUDED': 3, 'SIMULTANEOUS': 4, 'NONE': 5}