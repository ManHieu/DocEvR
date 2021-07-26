import gc
import os
import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
from utils.constant import CUDA


class SentenceEncoder():
    def __init__(self, roberta_type) -> None:
        super().__init__()
        self.roberta_type = roberta_type
        if os.path.exists("./pretrained_models/models/{}".format(roberta_type)):
            encoder = AutoModel.from_pretrained("./pretrained_models/models/{}".format(roberta_type), output_hidden_states=True)
        else:
            encoder = AutoModel.from_pretrained(roberta_type, output_hidden_states=True)
        if CUDA:
            self.encoder = encoder.cuda()
        else:
            self.encoder = encoder
        self.encoder.eval()
    
    def encode(self, sentence, mask=None, is_ctx=False):
        sentence = torch.tensor(sentence)
        if len(sentence.size()) == 1:
            sentence = sentence.unsqueeze(0)
        if mask != None:
            mask = torch.tensor(mask)
            if len(mask.size()) == 1:
                mask = mask.unsqueeze(0)
        if sentence.size(0) <= 1500:
            if CUDA:
                if mask != None:
                    mask = mask.cuda()
                sentence = sentence.cuda()
            with torch.no_grad():
                s_encoder = self.encoder(sentence, mask)[0].data.cpu()
                self.encoder.zero_grad(set_to_none=True)
            if is_ctx==True:
                s_encoder = s_encoder[:, 0]
            return s_encoder # ns x s_len x 768
        
        if sentence.size(0) > 1500:
            n = sentence.size(0)//1500
            presents = []
            for i in range(1, n+1):
                start = (i-1) * 1500
                end = i * 1500
                sent = sentence[start:end, :]
                mk = mask[start:end, :]
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
                presents = [torch.cat(presents, dim=0)]
                del s_encoder
                gc.collect()
                # print(len(presents))
            start = n * 1500
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

        
