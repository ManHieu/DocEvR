import math
import pdb
import numpy as np
np.random.seed(1741)
import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import os
import torch.nn as nn
from transformers import AutoModel
from utils.constant import CUDA
from utils.tools import padding

class SentenceEncoder():
    def __init__(self, roberta_type) -> None:
        super().__init__()
        self.roberta_type = roberta_type
        if os.path.exists("/vinai/hieumdt/pretrained_models/models/{}".format(roberta_type)):
            encoder = AutoModel.from_pretrained("/vinai/hieumdt/pretrained_models/models/{}".format(roberta_type), output_hidden_states=True)
        else:
            encoder = AutoModel.from_pretrained(roberta_type, output_hidden_states=True)
        if CUDA:
            self.encoder = encoder.cuda()
        else:
            self.encoder = encoder
        self.encoder.eval()
    
    def encode(self, sentence, mask=None, is_ctx=False):
        if type(sentence[0]) == list:
            max_len = max([len(sent) for sent in sentence])
            sentence= [padding(sent, max_sent_len=max_len)[0] for sent in sentence]
            mask = [padding(m, max_sent_len=max_len, pos=True) for m in mask]
        
        sentence = torch.tensor(sentence, dtype=torch.long)
        if len(sentence.size()) == 1:
            sentence = sentence.unsqueeze(0)
        if mask != None:
            mask = torch.tensor(mask, dtype=torch.long)
            if len(mask.size()) == 1:
                mask = mask.unsqueeze(0)
        if sentence.size(0) <= 1200:
            if CUDA:
                if mask != None:
                    mask = mask.cuda()
                sentence = sentence.cuda()
            with torch.no_grad():
                num_para = math.ceil(sentence.size(1) / 512.0)
                s_encoder = []
                for i in range(num_para):
                    if mask != None:
                        para_mask = mask[:, i*512:(i+1)*512]
                    else:
                        para_mask = mask
                    s_encoder.append(self.encoder(sentence[:, i*512:(i+1)*512], para_mask)[0].data.cpu())
                    self.encoder.zero_grad(set_to_none=True)
                s_encoder = torch.cat(s_encoder, dim=1)
            if is_ctx==True:
                s_encoder = s_encoder[:, 0]
            return s_encoder # ns x s_len x 768
        
        if sentence.size(0) > 1200:
            n = sentence.size(0)//1200
            presents = []
            for i in range(1, n+1):
                start = (i-1) * 1200
                end = i * 1200
                sent = sentence[start:end, :]
                mk = mask[start:end, :]
                if CUDA:
                    mk = mk.cuda()
                    sent = sent.cuda()
                with torch.no_grad():
                    num_para = math.ceil(sentence.size(1) / 512.0)
                    s_encoder = []
                    for i in range(num_para):
                        if mask != None:
                            para_mask = mask[:, i*512:(i+1)*512]
                        else:
                            para_mask = mask
                        s_encoder.append(self.encoder(sentence[:, i*512:(i+1)*512], para_mask)[0].data.cpu())
                        self.encoder.zero_grad(set_to_none=True)
                    s_encoder = torch.cat(s_encoder, dim=1)
                if is_ctx==True:
                    presents.append(s_encoder[:, 0])
                else:
                    presents.append(s_encoder)
                presents = [torch.cat(presents, dim=0)]
                # print(len(presents))
            start = n * 1200
            sent = sentence[start:, :]
            mk = mask[start:, :]
            if CUDA:
                mk = mk.cuda()
                sent = sent.cuda()
            with torch.no_grad():
                s_encoder = self.encoder(sent, mk)[0].detach().cpu()
                self.encoder.zero_grad(set_to_none=True)
            if is_ctx==True:
                presents.append(s_encoder[:, 0])
            else:
                presents.append(s_encoder)
            # print(presents)
            presents = torch.cat(presents, dim=0)
            # assert presents.size(0) == sentence.size(0)
            return presents # ns x s_len x 768

        

