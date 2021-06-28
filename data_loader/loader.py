from models.encode_augm_sent_model import SentenceEncoder
import os
from utils.constant import CUDA
import tqdm
import random
import torch
from itertools import combinations
from data_loader.reader import i2b2_xml_reader, tbd_tml_reader, tml_reader, tsvx_reader
from utils.tools import create_target, padding, pos_to_id
from sklearn.model_selection import train_test_split
from models.encode_augm_sent_model import SentenceEncoder
from transformers import AutoModel
import torch.nn as nn


sent_encoder = SentenceEncoder('roberta-base')
if CUDA:
    sent_encoder = sent_encoder.cuda()


class Reader(object):
    def __init__(self, type) -> None:
        super().__init__()
        self.type = type
    
    def read(self, dir_name, file_name):
        if self.type == 'tsvx':
            return tsvx_reader(dir_name, file_name)
        elif self.type == 'tml':
            return tml_reader(dir_name, file_name)
        elif self.type == 'i2b2_xml':
            return i2b2_xml_reader(dir_name, file_name)
        elif self.type == 'tbd_tml':
            return tbd_tml_reader(dir_name, file_name)
        else:
            raise ValueError("We have not supported {} type yet!".format(self.type))

def load_dataset(dir_name, type):
    reader = Reader(type)
    onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    corpus = []
    i = 0
    for file_name in tqdm.tqdm(onlyfiles):
        if i == 5:
            break
        i = i + 1
        if type == 'i2b2_xml':
            if file_name.endswith('.xml'):
                my_dict = reader.read(dir_name, file_name)
                if my_dict != None:
                    corpus.append(my_dict)
        else:
            my_dict = reader.read(dir_name, file_name)
            if my_dict != None:
                corpus.append(my_dict)
    return corpus


def loader(dataset, min_ns):
    def get_data_point(my_dict, flag):
        data = []
        short_data = []
        eids = my_dict['event_dict'].keys()
        pair_events = list(combinations(eids, 2))

        ctx_id_augm = []
        for pair in pair_events:
            x, y = pair
            
            x_sent_id = my_dict['event_dict'][x]['sent_id']
            y_sent_id = my_dict['event_dict'][y]['sent_id']
            ctx_id = list(range(len( my_dict["sentences"])))
            ctx_id.remove(x_sent_id)
            ctx_id.remove(y_sent_id)
            id_augm = [sorted(x_sent_id, y_sent_id, id) for id in ctx_id]
            ctx_id_augm.extend(id_augm)
        ctx_id_augm = list(set(ctx_id_augm))

        ctx_augm = []
        ctx_augm_mask = []
        for ids in ctx_id_augm:
            ids = set(ids)
            sent = []
            for id in ids:
                sent = sent + my_dict["sentences"][id]["roberta_subword_to_ID"][1:]
            sent = [0] + sent
            pad, mask = padding(sent, max_sent_len=256)
            ctx_augm.append(pad)
            ctx_augm_mask.append(mask)
        _augm_emb = sent_encoder(ctx_augm, ctx_augm_mask)
        if CUDA:
            _augm_emb = _augm_emb.cpu()
        _ctx_augm_emb = {}
        for i in range(len(ctx_id_augm)):
            print(_augm_emb[i].size())
            _ctx_augm_emb[ctx_id_augm[i]] = _augm_emb[i]
        
        for pair in pair_events:
            x, y = pair
            
            x_sent_id = my_dict['event_dict'][x]['sent_id']
            y_sent_id = my_dict['event_dict'][y]['sent_id']

            x_sent = my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"]
            y_sent = my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"]

            x_position = my_dict["event_dict"][x]["roberta_subword_id"]
            y_position = my_dict["event_dict"][y]["roberta_subword_id"]

            x_sent_pos = pos_to_id(my_dict["sentences"][x_sent_id]["roberta_subword_pos"])
            y_sent_pos = pos_to_id(my_dict["sentences"][y_sent_id]["roberta_subword_pos"])
            
            target = create_target(x_sent, y_sent, x_sent_id, y_sent_id)
            target_emb = sent_encoder(target_emb).squeeze()
            target_len = len(target)
            if CUDA:
                target_emb = target_emb.cpu()

            ctx = []
            ctx_emb = []
            ctx_pos = []
            ctx_len = []
            ctx_id = list(range(len( my_dict["sentences"])))
            ctx_id.remove(x_sent_id)
            ctx_id.remove(y_sent_id)
            for sent_id in ctx_id:
                sent = my_dict["sentences"][sent_id]['roberta_subword_to_ID']
                sent_pos = pos_to_id(my_dict["sentences"][sent_id]['roberta_subword_pos'])
                ctx.append(sent)
                ctx_pos.append(sent_pos)
                ctx_len.append(len(sent))
                sent_emb = _ctx_augm_emb[sorted(x_sent_id, y_sent_id, sent_id)]
                ctx_emb.append(sent_emb)
            ctx_emb = torch.stack(ctx_emb, dim=0) # ns x 768
            print(ctx_emb.size())

            xy = my_dict["relation_dict"].get((x, y))
            yx = my_dict["relation_dict"].get((y, x))

            candidates = [
                [str(x), str(y), x_sent, y_sent, x_sent_id, y_sent_id, x_sent_pos, y_sent_pos, x_position, y_position, 
                ctx_id, target, target_emb, target_len, ctx, ctx_emb, ctx_len, ctx_pos, flag, xy],
                [str(y), str(x), y_sent, x_sent, y_sent_id, x_sent_id, y_sent_pos, x_sent_pos, y_position, x_position, 
                ctx_id, target, target_emb, target_len, ctx, ctx_emb, ctx_len, ctx_pos, flag, yx],
            ]
            for item in candidates:
                if item[-1] != None and len(ctx_len) >= min_ns:
                    data.append(item)
                if item[-1] != None and len(ctx_len) < min_ns:
                    short_data.append(item)
        return data, short_data
           
    train_set = []
    train_short = []
    test_set = []
    test_short = []
    validate_set = []
    validate_short = []
    if dataset == "MATRES":
        print("MATRES Loading .......")
        aquaint_dir_name = "./datasets/MATRES/TBAQ-cleaned/AQUAINT/"
        timebank_dir_name = "./datasets/MATRES/TBAQ-cleaned/TimeBank/"
        platinum_dir_name = "./datasets/MATRES/te3-platinum/"
        validate = load_dataset(aquaint_dir_name, 'tml')
        train = load_dataset(timebank_dir_name, 'tml')
        test = load_dataset(platinum_dir_name, 'tml')
        train, validate = train_test_split(train + validate, test_size=0.2, train_size=0.8)
        
        for my_dict in tqdm.tqdm(train):
            data, short_data = get_data_point(my_dict, 2)
            train_set.extend(data)
            train_short.extend(short_data)
        for my_dict in tqdm.tqdm(test):
            data, short_data = get_data_point(my_dict, 2)
            test_set.extend(data)
            test_short.extend(short_data)
        for my_dict in tqdm.tqdm(validate):
            data, short_data = get_data_point(my_dict, 2)
            validate_set.extend(data)
            validate_short.extend(short_data)
        print("Train_size: {}".format(len(train_set)))
        print("Test_size: {}".format(len(test_set)))
        print("Validate_size: {}".format(len(validate_set)))

    if dataset == "HiEve":
        print("HiEve Loading .....")
        dir_name = "./datasets/hievents_v2/processed/"
        corpus = load_dataset(dir_name, 'tsvx')
        train, test = train_test_split(corpus, train_size=0.8, test_size=0.2)
        train, validate = train_test_split(train, train_size=0.75, test_size=0.25)
        sample = 0.015
        for my_dict in tqdm.tqdm(train):
            data, short_data = get_data_point(my_dict, 1)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        train_set.append(item)
                else:
                    train_set.append(item)
            for item in short_data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        train_short.append(item)
                else:
                    train_short.append(item)
        for my_dict in tqdm.tqdm(test):
            data, short_data = get_data_point(my_dict, 1)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < 0.015:
                        test_set.append(item)
                else:
                    test_set.append(item)
            for item in short_data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        test_short.append(item)
                else:
                    test_short.append(item)
        for my_dict in tqdm.tqdm(validate):
            data, short_data = get_data_point(my_dict, 1)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        validate_set.append(item)
                else:
                    validate_set.append(item)
            for item in short_data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        validate_short.append(item)
                else:
                    validate_short.append(item)
        print("Train_size: {}".format(len(train_set)))
        print("Test_size: {}".format(len(test_set)))
        print("Validate_size: {}".format(len(validate_set)))

    if dataset == "I2B2":
        print("I2B2 Loading .....")
        dir_name = "./datasets/i2b2_2012/2012-07-15.original-annotation.release/"
        corpus = load_dataset(dir_name, 'i2b2_xml')
        train, test = train_test_split(corpus, train_size=0.8, test_size=0.2)
        train, validate = train_test_split(train, train_size=0.75, test_size=0.25)
        for my_dict in tqdm.tqdm(train):
            data, short_data = get_data_point(my_dict, 3)
            train_set.extend(data)
            train_short.extend(short_data)
        for my_dict in tqdm.tqdm(test):
            data, short_data = get_data_point(my_dict, 3)
            test_set.extend(data)
            test_short.extend(short_data)
        for my_dict in tqdm.tqdm(validate):
            data, short_data = get_data_point(my_dict, 3)
            validate_set.extend(data)
            validate_short.extend(short_data)
        print("Train_size: {}".format(len(train_set)))
        print("Test_size: {}".format(len(test_set)))
        print("Validate_size: {}".format(len(validate_set)))

    if dataset == 'TBD':
        print("Timebank Dense Loading .....")
        train_dir = "./datasets/TimeBank-dense/train/"
        test_dir = "./datasets/TimeBank-dense/test/"
        validate_dir = "./datasets/TimeBank-dense/dev/"
        train = load_dataset(train_dir, 'tbd_tml')
        test = load_dataset(test_dir, 'tbd_tml')
        validate = load_dataset(validate_dir, 'tbd_tml')
        for my_dict in tqdm.tqdm(train):
            data, short_data = get_data_point(my_dict, 4)
            train_set.extend(data)
            train_short.extend(short_data)
        for my_dict in tqdm.tqdm(test):
            data, short_data = get_data_point(my_dict, 4)
            test_set.extend(data)
            test_short.extend(short_data)
        for my_dict in tqdm.tqdm(validate):
            data, short_data = get_data_point(my_dict, 4)
            validate_set.extend(data)
            validate_short.extend(short_data)
        print("Train_size: {}".format(len(train_set)))
        print("Test_size: {}".format(len(test_set)))
        print("Validate_size: {}".format(len(validate_set)))
        print("Train_size: {}".format(len(train_short)))
        print("Test_size: {}".format(len(test_short)))
        print("Validate_size: {}".format(len(validate_short)))

    return train_set, test_set, validate_set, train_short, test_short, validate_short
        