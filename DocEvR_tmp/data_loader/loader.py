import os
import tqdm
import random
from itertools import combinations
from data_loader.reader import i2b2_xml_reader, tbd_tml_reader, tml_reader, tsvx_reader
from utils.tools import augment_ctx, pos_to_id


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
            x_position = my_dict["event_dict"][x]["roberta_subword_id"]
            y_position = my_dict["event_dict"][y]["roberta_subword_id"]
            x_sent_pos = pos_to_id(my_dict["sentences"][x_sent_id]["roberta_subword_pos"])
            y_sent_pos = pos_to_id(my_dict["sentences"][y_sent_id]["roberta_subword_pos"])

            x_ctx = []
            x_ctx_augm = []
            x_ctx_pos = []
            x_ctx_len = []
            y_ctx = []
            y_ctx_augm = []
            y_ctx_pos = []
            y_ctx_len = []
            for sent_id in range(len(my_dict["sentences"])):
                if sent_id != x_sent_id:
                    sent = my_dict["sentences"][sent_id]['roberta_subword_to_ID']
                    sent_augm = augment_ctx(x_sent, x_sent_id, sent, sent_id)
                    sent_pos = pos_to_id(my_dict["sentences"][sent_id]['roberta_subword_pos'])
                    x_ctx.append(sent)
                    x_ctx_augm.append(sent_augm)
                    x_ctx_pos.append(sent_pos)
                    x_ctx_len.append(len(sent))

                if sent_id != y_sent_id:
                    sent = my_dict["sentences"][sent_id]['roberta_subword_to_ID']
                    sent_augm = augment_ctx(y_sent, y_sent_id, sent, sent_id)
                    sent_pos = pos_to_id(my_dict["sentences"][sent_id]['roberta_subword_pos'])
                    y_ctx.append(sent)
                    y_ctx_augm.append(sent_augm)
                    y_ctx_pos.append(sent_pos)
                    y_ctx_len.append(len(sent))
            
            xy = my_dict["relation_dict"].get((x, y))
            yx = my_dict["relation_dict"].get((y, x))
            
            candidates = [
                [x_sent_id, y_sent_id, x_sent, y_sent, x_position, y_position, x_sent_pos, y_sent_pos, x_ctx, y_ctx, 
                x_ctx_len, y_ctx_len, x_ctx_augm, y_ctx_augm, x_ctx_pos, y_ctx_pos, flag, xy],
                [y_sent_id, x_sent_id, y_sent, x_sent, y_position, x_position, y_sent_pos, x_sent_pos, y_ctx, x_ctx, 
                y_ctx_len, x_ctx_len, y_ctx_augm, x_ctx_augm, y_ctx_pos, x_ctx_pos, flag, yx],
            ]

            for item in candidates:
                if item[-1] != None and len(x_ctx_len) == len(y_ctx_len) and len(x_ctx_len) >= min_ns:
                    data.append(item)
                if len(x_ctx_len) < min_ns:
                    short_data.append(item)
        return data, short_data

