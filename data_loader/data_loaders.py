from itertools import combinations
import os
import tqdm
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_loader.EventDataset import EventDataset
from data_loader.document_reader import tsvx_reader, tml_reader, i2b2_xml_reader, tbd_tml_reader
from utils.tools import *


class Reader():
    def __init__(self, type) -> None:
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
            raise ValueError("Wrong reader!")


def load_dataset(dir_name, type):
    reader = Reader(type)
    onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    corpus = []

    for file_name in tqdm.tqdm(onlyfiles):
        if type == 'i2b2_xml':
            if file_name.endswith('.xml'):
                my_dict = reader.read(dir_name, file_name)
                if my_dict != None:
                    corpus.append(my_dict)
        else:
            my_dict = reader.read(dir_name, file_name)
            if my_dict != None:
                corpus.append(my_dict)
        
    # train_set, test_set = train_test_split(corpus, train_size=0.8, test_size=0.2)
    # train_set, validate_set = train_test_split(train_set, train_size=0.75, test_size=0.25)
    # print("Train size {}".format(len(train_set)))
    # print("Test size {}".format(len(test_set)))
    # print("Validate size {}".format(len(validate_set)))
    return corpus

def single_loader(dataset):
    def get_data_point(my_dict, flag):
        data = []
        eids = my_dict['event_dict'].keys()
        pair_events = list(combinations(eids, 2))
        for pair in pair_events:
            x, y = pair
            x_sent_id = my_dict['event_dict'][x]['sent_id']
            y_sent_id = my_dict['event_dict'][y]['sent_id']

            x_sent = padding(my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"])
            y_sent = padding(my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"])

            x_position = my_dict["event_dict"][x]["roberta_subword_id"]
            y_position = my_dict["event_dict"][y]["roberta_subword_id"]

            x_sent_pos = pos_to_id(padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True))
            y_sent_pos = pos_to_id(padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True))

            xy = my_dict["relation_dict"].get((x, y))
            yx = my_dict["relation_dict"].get((y, x))
            candidates = [[str(x), str(y), x_sent, y_sent, x_position, y_position, x_sent_pos, y_sent_pos, flag, xy],
                        [str(y), str(x), y_sent, x_sent, y_position, x_position, y_sent_pos, x_sent_pos, flag, yx]]
            for item in candidates:
                if item[-1] != None:
                    data.append(item)
        return data

    train_set = []
    test_set = []
    validate_set = []
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
            data = get_data_point(my_dict, 2)
            train_set.extend(data)
        for my_dict in tqdm.tqdm(test):
            data = get_data_point(my_dict, 2)
            test_set.extend(data)
        for my_dict in tqdm.tqdm(validate):
            data = get_data_point(my_dict, 2)
            validate_set.extend(data)
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
            data = get_data_point(my_dict, 1)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        train_set.append(item)
                else:
                    train_set.append(item)
        for my_dict in tqdm.tqdm(test):
            data = get_data_point(my_dict, 1)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < 0.015:
                        test_set.append(item)
                else:
                    test_set.append(item)
        for my_dict in tqdm.tqdm(validate):
            data = get_data_point(my_dict, 1)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        validate_set.append(item)
                else:
                    validate_set.append(item)
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
            data = get_data_point(my_dict, 3)
            train_set.extend(data)
        for my_dict in tqdm.tqdm(test):
            data = get_data_point(my_dict, 3)
            test_set.extend(data)
        for my_dict in tqdm.tqdm(validate):
            data = get_data_point(my_dict, 3)
            validate_set.extend(data)
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
            data = get_data_point(my_dict, 4)
            train_set.extend(data)
        for my_dict in tqdm.tqdm(test):
            data = get_data_point(my_dict, 4)
            test_set.extend(data)
        for my_dict in tqdm.tqdm(validate):
            data = get_data_point(my_dict, 4)
            validate_set.extend(data)
        print("Train_size: {}".format(len(train_set)))
        print("Test_size: {}".format(len(test_set)))
        print("Validate_size: {}".format(len(validate_set)))

    # train_loader = DataLoader(EventDataset(train_set), batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(EventDataset(test_set), batch_size=batch_size, shuffle=True)
    # validate_loader = DataLoader(EventDataset(validate_set), batch_size=batch_size, shuffle=True)
    return train_set, test_set, validate_set