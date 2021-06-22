import os
from utils.constant import CUDA
import tqdm
import random
import torch
from itertools import combinations
from data_loader.reader import i2b2_xml_reader, tbd_tml_reader, tml_reader, tsvx_reader
from utils.tools import augment_ctx, padding, pos_to_id
from sklearn.model_selection import train_test_split
from transformers import AutoModel


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


class SentenceEncoder(object):
    def __init__(self, roberta_type) -> None:
        super().__init__()
        self.roberta_type = roberta_type
        if os.path.exists("./pretrained_models/models/{}".format(roberta_type)):
            print("Loading pretrain model from local ......")
            self.encoder = AutoModel.from_pretrained("./pretrained_models/models/{}".format(roberta_type), output_hidden_states=True)
        else:
            print("Loading pretrain model ......")
            self.encoder = AutoModel.from_pretrained(roberta_type, output_hidden_states=True)
        if CUDA:
            self.encoder = self.encoder.cuda()
    
    def encode(self, sentence):
        sentence = torch.tensor(sentence, dtype=torch.long)
        if CUDA:
            sentence = sentence.cuda()
        with torch.no_grad():
            s_encoder = self.encoder(sentence.unsqueeze(0))[0]
        # print(s_encoder)
        return s_encoder[:, 0] # 1 x 768


sent_encoder = SentenceEncoder('roberta-base')

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
        for pair in pair_events:
            x, y = pair
            x_sent_id = my_dict['event_dict'][x]['sent_id']
            y_sent_id = my_dict['event_dict'][y]['sent_id']
            x_sent = my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"]
            y_sent = my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"]
            x_sent_len = len(x_sent)
            y_sent_len = len(y_sent)
            x_sent_emb = sent_encoder.encode(x_sent).squeeze()
            y_sent_emb = sent_encoder.encode(y_sent).squeeze()
            x_position = my_dict["event_dict"][x]["roberta_subword_id"]
            y_position = my_dict["event_dict"][y]["roberta_subword_id"]
            x_sent_pos = pos_to_id(my_dict["sentences"][x_sent_id]["roberta_subword_pos"])
            y_sent_pos = pos_to_id(my_dict["sentences"][y_sent_id]["roberta_subword_pos"])

            x_ctx = []
            x_ctx_augm = []
            x_ctx_augm_emb = []
            x_ctx_pos = []
            x_ctx_len = []
            y_ctx = []
            y_ctx_augm = []
            y_ctx_augm_emb = []
            y_ctx_pos = []
            y_ctx_len = []
            for sent_id in range(len(my_dict["sentences"])):
                if sent_id != x_sent_id:
                    sent = my_dict["sentences"][sent_id]['roberta_subword_to_ID']
                    sent_augm = padding(augment_ctx(x_sent, x_sent_id, sent, sent_id))
                    # sent_augm_emb = sent_encoder.encode(sent_augm)
                    sent_pos = pos_to_id(my_dict["sentences"][sent_id]['roberta_subword_pos'])
                    x_ctx.append(sent)
                    x_ctx_augm.append(sent_augm)
                    # x_ctx_augm_emb.append(sent_augm_emb)
                    x_ctx_pos.append(sent_pos)
                    x_ctx_len.append(len(sent))

                if sent_id != y_sent_id:
                    sent = my_dict["sentences"][sent_id]['roberta_subword_to_ID']
                    sent_augm = padding(augment_ctx(y_sent, y_sent_id, sent, sent_id))
                    # sent_augm_emb = sent_encoder.encode(sent_augm)
                    sent_pos = pos_to_id(my_dict["sentences"][sent_id]['roberta_subword_pos'])
                    y_ctx.append(sent)
                    y_ctx_augm.append(sent_augm)
                    # y_ctx_augm_emb.append(sent_augm_emb)
                    y_ctx_pos.append(sent_pos)
                    y_ctx_len.append(len(sent))
            
            x_ctx_augm_emb = sent_encoder.encode(x_ctx_augm)
            y_ctx_augm_emb = sent_encoder.encode(y_ctx_augm)
            xy = my_dict["relation_dict"].get((x, y))
            yx = my_dict["relation_dict"].get((y, x))
            
            candidates = [
                [x_sent_id, y_sent_id, x_sent, y_sent, x_sent_len, y_sent_len, x_sent_emb, y_sent_emb, x_position, y_position, x_sent_pos, y_sent_pos, 
                x_ctx, y_ctx, x_ctx_len, y_ctx_len, x_ctx_augm, y_ctx_augm, x_ctx_augm_emb, y_ctx_augm_emb, x_ctx_pos, y_ctx_pos, flag, xy],
                [y_sent_id, x_sent_id, y_sent, x_sent, y_sent_len, x_sent_len, y_sent_emb, x_sent_emb,  y_position, x_position, y_sent_pos, x_sent_pos, 
                y_ctx, x_ctx, y_ctx_len, x_ctx_len, y_ctx_augm, x_ctx_augm, y_ctx_augm_emb, x_ctx_augm_emb, y_ctx_pos, x_ctx_pos, flag, yx],
            ]

            for item in candidates:
                if item[-1] != None and len(x_ctx_len) == len(y_ctx_len) and len(x_ctx_len) >= min_ns:
                    data.append(item)
                if item[-1] != None and len(x_ctx_len) < min_ns:
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
        