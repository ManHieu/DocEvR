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

def joint_constrained_loader(dataset, downsample, batch_size):
    def get_data_train(my_dict):
        train_data = []
        eids = my_dict['event_dict'].keys()
        triplets = list(combinations(eids, 3))
        for triplet in triplets:
            x, y, z = triplet

            x_sent_id = my_dict['event_dict'][x]['sent_id']
            y_sent_id = my_dict['event_dict'][y]['sent_id']
            z_sent_id = my_dict['event_dict'][z]['sent_id']

            x_sent = padding(my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"])
            y_sent = padding(my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"])
            z_sent = padding(my_dict["sentences"][z_sent_id]["roberta_subword_to_ID"])

            x_position = my_dict["event_dict"][x]["roberta_subword_id"]
            y_position = my_dict["event_dict"][y]["roberta_subword_id"]
            z_position = my_dict["event_dict"][z]["roberta_subword_id"]

            x_sent_pos = padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True)
            y_sent_pos = padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True)
            z_sent_pos = padding(my_dict["sentences"][z_sent_id]["roberta_subword_pos"], pos = True)

            xy = my_dict["relation_dict"].get((x, y))
            yx = my_dict["relation_dict"].get((y, x))
            yz = my_dict["relation_dict"].get((y, z))
            zy = my_dict["relation_dict"].get((z, y))
            xz = my_dict["relation_dict"].get((x, z))
            zx = my_dict["relation_dict"].get((z, x))

            candidates = [[str(x), str(y), str(z), x_sent, y_sent, z_sent, x_position, y_position, z_position, x_sent_pos, y_sent_pos, z_sent_pos, xy, yz, xz],
                        [str(y), str(x), str(z), y_sent, x_sent, z_sent, y_position, x_position, z_position, y_sent_pos, x_sent_pos, z_sent_pos, yx, xz, yz],
                        [str(y), str(z), str(x), y_sent, z_sent, x_sent, y_position, z_position, x_position, y_sent_pos, z_sent_pos, x_sent_pos, yz, zx, yx],
                        [str(z), str(y), str(x), z_sent, y_sent, x_sent, z_position, y_position, x_position, z_sent_pos, y_sent_pos, x_sent_pos, zy, yx, zx],
                        [str(x), str(z), str(y), x_sent, z_sent, y_sent, x_position, z_position, y_position, x_sent_pos, z_sent_pos, y_sent_pos, xz, zy, xy],
                        [str(z), str(x), str(y), z_sent, x_sent, y_sent, z_position, x_position, y_position, z_sent_pos, x_sent_pos, y_sent_pos, zx, xy, zy]]
            
            for item in candidates:
                if None not in item[-3:]:
                    train_data.append(item)
        return train_data

    def get_data_test(my_dict):
        test_data = []
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

            x_sent_pos = padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True)
            y_sent_pos = padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True)

            xy = my_dict["relation_dict"].get((x, y))
            yx = my_dict["relation_dict"].get((y, x))
            candidates = [[str(x), str(y), str(x), x_sent, y_sent, x_sent, x_position, y_position, x_position, x_sent_pos, y_sent_pos, x_sent_pos, xy, xy, xy],
                        [str(y), str(x), str(y), y_sent, x_sent, y_sent, y_position, x_position, y_position, y_sent_pos, x_sent_pos, y_sent_pos, yx, yx, yx]]
            for item in candidates:
                if None not in item[-3:]:
                    test_data.append(item)
        return test_data

    train_set_HIEVE = []
    valid_set_HIEVE = []
    test_set_HIEVE = []
    train_set_MATRES = []
    valid_set_MATRES = []
    test_set_MATRES = []
    train_set_I2B2 = []
    valid_set_I2B2 = []
    test_set_I2B2 = []
    if dataset in ["HiEve", "Joint"]:
        # ========================
        #       HiEve Dataset
        # ========================
        print("HiEve Loading .....")
        dir_name = "./datasets/hievents_v2/processed/"
        corpus = load_dataset(dir_name, 'tsvx')
        train, test = train_test_split(corpus, train_size=0.8, test_size=0.2)
        train, validate = train_test_split(train, train_size=0.75, test_size=0.25)
        undersmp_ratio = 0.4
        print("Loading train data.....")
        for my_dict in tqdm.tqdm(train):
            train_data = get_data_train(my_dict)
            for item in train_data:
                if item[-3]==3 and item[-2]==3:
                    pass
                elif 3 in item[-3:]:
                    if random.uniform(0, 1) < downsample:
                        item.append(0) # 0 is HiEve
                        train_set_HIEVE.append(item)
                else:
                    item.append(0)
                    train_set_HIEVE.append(item)
        print("Number data points: {}".format(len(train_set_HIEVE)))
        
        print("Loading test data.....")
        for my_dict in tqdm.tqdm(test):
            test_data = get_data_test(my_dict)
            for item in test_data:
                if item[-3]==3:
                    if random.uniform(0, 1) < undersmp_ratio:
                        item.append(0)
                        test_set_HIEVE.append(item)
                else:
                    item.append(0)
                    test_set_HIEVE.append(item)
        print("Number data points: {}".format(len(test_set_HIEVE)))
        
        print("Loading validate data ....")
        for my_dict in tqdm.tqdm(validate):
            valid_data = get_data_test(my_dict)
            for item in valid_data:
                if item[-3]==3:
                    if random.uniform(0, 1) < undersmp_ratio:
                        item.append(0)
                        valid_set_HIEVE.append(item)
                else:
                    item.append(0)
                    valid_set_HIEVE.append(item)
        print("Number data points: {}".format(len(valid_set_HIEVE)))

    if dataset in ["MATRES", "Joint"]:
        # ========================
        #       MATRES Dataset
        # ========================
        print("MATRES Loading .....")
        aquaint_dir_name = "./datasets/MATRES/TBAQ-cleaned/AQUAINT/"
        timebank_dir_name = "./datasets/MATRES/TBAQ-cleaned/TimeBank/"
        platinum_dir_name = "./datasets/MATRES/te3-platinum/"
        validate = load_dataset(aquaint_dir_name, 'tml')
        train = load_dataset(timebank_dir_name, 'tml')
        test = load_dataset(platinum_dir_name, 'tml')
        
        print("Loading train data.....")
        for my_dict in tqdm.tqdm(train):
            if my_dict != None:
                train_data = get_data_train(my_dict)
                for item in train_data:
                    item.append(1) # 1 is MATRES
                    train_set_MATRES.append(item)
        print("Number data points: {}".format(len(train_set_MATRES)))

        print("Loading test data.....")
        for my_dict in tqdm.tqdm(test):
            if my_dict != None:
                test_data = get_data_test(my_dict)
                for item in test_data:
                    item.append(1) # 1 is MATRES
                    test_set_MATRES.append(item)
        print("Number data points: {}".format(len(test_set_MATRES)))

        print("Loading validate data ....")
        for my_dict in tqdm.tqdm(validate):
            if my_dict != None:
                validate_data = get_data_test(my_dict)
                for item in validate_data:
                    item.append(1) # 1 is MATRES
                valid_set_MATRES.append(item)
        print("Number data points: {}".format(len(valid_set_MATRES)))

    if dataset in ["I2B2", "Joint"]:
        # ========================
        #       I2B2 Dataset
        # ========================
        print("I2B2 Loading .....")
        dir_name = "./datasets/i2b2_2012/2012-07-15.original-annotation.release/"
        corpus = load_dataset(dir_name, 'i2b2_xml')
        train, test = train_test_split(corpus, train_size=0.8, test_size=0.2)
        train, validate = train_test_split(train, train_size=0.75, test_size=0.25)
        print("Loading test data.....")
        for my_dict in tqdm.tqdm(test):
            test_data = get_data_test(my_dict)
            for item in test_data:
                item.append(2) # 2 is I2B2
                test_set_I2B2.append(item)
        print("Number data points: {}".format(len(test_set_I2B2)))
        
        print("Loading validate data ....")
        for my_dict in tqdm.tqdm(validate):
            validate_data = get_data_test(my_dict)
            for item in validate_data:
                item.append(2) # 2 is I2B2
                valid_set_I2B2.append(item)
        print("Number data points: {}".format(len(valid_set_I2B2)))
        
        print("Loading train data.....")
        for my_dict in tqdm.tqdm(train):
            train_data = get_data_train(my_dict)
            for item in train_data:
                item.append(2) # 2 is I2B2
                train_set_I2B2.append(item)
        print("Number data points: {}".format(len(train_set_I2B2)))
 
    # ==============================================================
    #      Use DataLoader to convert to Pytorch acceptable form
    # ==============================================================
    if dataset == "MATRES":
        num_classes = 4
        train_dataloader_MATRES = DataLoader(EventDataset(train_set_MATRES), batch_size=batch_size, shuffle = True)
        valid_dataloader_MATRES = DataLoader(EventDataset(valid_set_MATRES), batch_size=batch_size, shuffle = True)    
        test_dataloader_MATRES = DataLoader(EventDataset(test_set_MATRES), batch_size=batch_size, shuffle = True) 
        return train_dataloader_MATRES, valid_dataloader_MATRES, test_dataloader_MATRES, None, None, None, None, num_classes 
    elif dataset == "HiEve":
        num_classes = 4
        train_dataloader_HIEVE = DataLoader(EventDataset(train_set_HIEVE), batch_size=batch_size, shuffle = True)
        valid_dataloader_HIEVE = DataLoader(EventDataset(valid_set_HIEVE), batch_size=batch_size, shuffle = True)    
        test_dataloader_HIEVE = DataLoader(EventDataset(test_set_HIEVE), batch_size=batch_size, shuffle = True) 
        return train_dataloader_HIEVE, None, None, valid_dataloader_HIEVE, test_dataloader_HIEVE, None, None, num_classes
    if dataset == "I2B2":
        num_classes = 3
        print("valid_set_I2B2 size: {}".format(len(valid_set_I2B2)))
        print("test_set_I2B2 size: {}".format(len(test_set_I2B2)))
        train_dataloader_I2B2 = DataLoader(EventDataset(train_set_I2B2), batch_size=batch_size, shuffle = True)
        valid_dataloader_I2B2 = DataLoader(EventDataset(valid_set_I2B2), batch_size=batch_size, shuffle = True)    
        test_dataloader_I2B2 = DataLoader(EventDataset(test_set_I2B2), batch_size=batch_size, shuffle = True) 
        return train_dataloader_I2B2, None, None, None, None, valid_dataloader_I2B2, test_dataloader_I2B2, num_classes 
    elif dataset == "Joint":
        num_classes = 8
        train_set_HIEVE.extend(train_set_MATRES)
        train_set_HIEVE.extend(train_set_I2B2)
        print("Train size: {}".format(len(train_set_HIEVE)))
        train_dataloader = DataLoader(EventDataset(train_set_HIEVE), batch_size=batch_size, shuffle = True)
        valid_dataloader_MATRES = DataLoader(EventDataset(valid_set_MATRES), batch_size=batch_size, shuffle = True)    
        test_dataloader_MATRES = DataLoader(EventDataset(test_set_MATRES), batch_size=batch_size, shuffle = True)
        valid_dataloader_HIEVE = DataLoader(EventDataset(valid_set_HIEVE), batch_size=batch_size, shuffle=True)    
        test_dataloader_HIEVE = DataLoader(EventDataset(test_set_HIEVE), batch_size=batch_size, shuffle = True)
        print("valid_set_I2B2 size: {}".format(len(valid_set_I2B2)))
        print("test_set_I2B2 size: {}".format(len(test_set_I2B2)))
        valid_dataloader_I2B2 = DataLoader(EventDataset(valid_set_I2B2), batch_size=batch_size, shuffle = True)    
        test_dataloader_I2B2 = DataLoader(EventDataset(test_set_I2B2), batch_size=batch_size, shuffle = True) 
        return train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, valid_dataloader_HIEVE, test_dataloader_HIEVE, valid_dataloader_I2B2, test_dataloader_I2B2, num_classes
    else:
        raise ValueError("Currently not supporting this dataset! -_-'")

def single_loader(dataset, batch_size):
    def get_data_point(my_dict):
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

            x_sent_pos = padding(my_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos = True)
            y_sent_pos = padding(my_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos = True)

            xy = my_dict["relation_dict"].get((x, y))
            yx = my_dict["relation_dict"].get((y, x))
            candidates = [[str(x), str(y), x_sent, y_sent, x_position, y_position, x_sent_pos, y_sent_pos, xy],
                        [str(y), str(x), y_sent, x_sent, y_position, x_position, y_sent_pos, x_sent_pos, yx]]
            for item in candidates:
                if item[-1] != None:
                    data.append(item)
        return data

    train_set = []
    test_set = []
    validate_set = []
    num_class = 0
    if dataset == "MATRES":
        print("MATRES Loading .......")
        aquaint_dir_name = "./datasets/MATRES/TBAQ-cleaned/AQUAINT/"
        timebank_dir_name = "./datasets/MATRES/TBAQ-cleaned/TimeBank/"
        platinum_dir_name = "./datasets/MATRES/te3-platinum/"
        validate = load_dataset(aquaint_dir_name, 'tml')
        train = load_dataset(timebank_dir_name, 'tml')
        test = load_dataset(platinum_dir_name, 'tml')
        num_class = 4
        # train, validate = train_test_split(train + validate, test_size=0.2, train_size=0.8)
        
        for my_dict in tqdm.tqdm(train):
            data = get_data_point(my_dict)
            train_set.extend(data)
        for my_dict in tqdm.tqdm(test):
            data = get_data_point(my_dict)
            test_set.extend(data)
        for my_dict in tqdm.tqdm(validate):
            data = get_data_point(my_dict)
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
        sample = 0.04
        for my_dict in tqdm.tqdm(train):
            data = get_data_point(my_dict)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        train_set.append(item)
                else:
                    train_set.append(item)
        for my_dict in tqdm.tqdm(test):
            data = get_data_point(my_dict)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        test_set.append(item)
                else:
                    test_set.append(item)
        for my_dict in tqdm.tqdm(validate):
            data = get_data_point(my_dict)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        validate_set.append(item)
                else:
                    validate_set.append(item)
        print("Train_size: {}".format(len(train_set)))
        print("Test_size: {}".format(len(test_set)))
        print("Validate_size: {}".format(len(validate_set)))
        num_class = 4

    if dataset == "I2B2":
        print("I2B2 Loading .....")
        dir_name = "./datasets/i2b2_2012/2012-07-15.original-annotation.release/"
        corpus = load_dataset(dir_name, 'i2b2_xml')
        train, test = train_test_split(corpus, train_size=0.8, test_size=0.2)
        train, validate = train_test_split(train, train_size=0.75, test_size=0.25)
        for my_dict in tqdm.tqdm(train):
            data = get_data_point(my_dict)
            train_set.extend(data)
        for my_dict in tqdm.tqdm(test):
            data = get_data_point(my_dict)
            test_set.extend(data)
        for my_dict in tqdm.tqdm(validate):
            data = get_data_point(my_dict)
            validate_set.extend(data)
        print("Train_size: {}".format(len(train_set)))
        print("Test_size: {}".format(len(test_set)))
        print("Validate_size: {}".format(len(validate_set)))
        num_class = 3

    if dataset == 'TBD':
        print("Timebank Dense Loading .....")
        train_dir = "./datasets/TimeBank-dense/train/"
        test_dir = "./datasets/TimeBank-dense/test/"
        validate_dir = "./datasets/TimeBank-dense/dev/"
        train = load_dataset(train_dir, 'tbd_tml')
        test = load_dataset(test_dir, 'tbd_tml')
        validate = load_dataset(validate_dir, 'tbd_tml')
        for my_dict in tqdm.tqdm(train):
            data = get_data_point(my_dict)
            train_set.extend(data)
        for my_dict in tqdm.tqdm(test):
            data = get_data_point(my_dict)
            test_set.extend(data)
        for my_dict in tqdm.tqdm(validate):
            data = get_data_point(my_dict)
            validate_set.extend(data)
        print("Train_size: {}".format(len(train_set)))
        print("Test_size: {}".format(len(test_set)))
        print("Validate_size: {}".format(len(validate_set)))
        num_class = 6

    train_loader = DataLoader(EventDataset(train_set), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(EventDataset(test_set), batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(EventDataset(validate_set), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, validate_loader, num_class