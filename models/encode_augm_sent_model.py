import os
import torch
import torch.nn as nn
from transformers import AutoModel
from utils.constant import CUDA


class SentenceEncoder(nn.Module):
    def __init__(self, roberta_type) -> None:
        super().__init__()
        self.roberta_type = roberta_type
        if os.path.exists("./pretrained_models/models/{}".format(roberta_type)):
            # print("Loading pretrain model from local ......")
            self.encoder = AutoModel.from_pretrained("./pretrained_models/models/{}".format(roberta_type), output_hidden_states=True)
        else:
            # print("Loading pretrain model ......")
            self.encoder = AutoModel.from_pretrained(roberta_type, output_hidden_states=True)
    
    def forward(self, sentence, mask=None):
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
            # print("Sentence size: ", sentence.size())
            with torch.no_grad():
                s_encoder = self.encoder(sentence, mask)[0].cpu()
            return s_encoder[:, 0] # ns x 768
        
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
                    s_encoder = self.encoder(sent, mk)[0][:, 0].cpu()
                presents.append(s_encoder)
            start = n * 1200
            sent = sentence[start:, :]
            mk = mask[start:, :]
            if CUDA:
                mk = mk.cuda()
                sent = sent.cuda()
            with torch.no_grad():
                s_encoder = self.encoder(sent, mk)[0][:, 0].cpu()
            presents.append(s_encoder)
            presents = torch.cat(presents, dim=0)
            if CUDA:
                presents = presents
            # assert presents.size(0) == sentence.size(0)
            return presents

        

