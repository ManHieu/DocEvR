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
            print("Loading pretrain model from local ......")
            self.encoder = AutoModel.from_pretrained("./pretrained_models/models/{}".format(roberta_type), output_hidden_states=True)
        else:
            print("Loading pretrain model ......")
            self.encoder = AutoModel.from_pretrained(roberta_type, output_hidden_states=True)
    
    def forward(self, sentence, mask=None):
        sentence = torch.tensor(sentence, dtype=torch.long)
        if mask != None:
            mask = torch.tensor(mask, dtype=torch.long)
            if CUDA:
                mask = mask.cuda()
            if len(mask.size()) == 1:
                mask = mask.unsqueeze()
        if CUDA:
            sentence = sentence.cuda()
        # print("Sentence size: ", sentence.size())
        if len(sentence.size()) == 1:
            sentence = sentence.unsqueeze(0)
        with torch.no_grad():
            s_encoder = self.encoder(sentence, mask)[0]

        # if sentence.size(0) > 50:
        #     sentence1 = sentence[:50, :]
        #     sentence2 = sentence[50:, :]
        #     with torch.no_grad():
        #         s_encoder1 = self.encoder(sentence1)[0]
        #         s_encoder2 = self.encoder(sentence2)[0]
        #     return torch.cat([s_encoder1[:, 0], s_encoder2[:, 0]], dim=0).cpu()
        # print(s_encoder)
        return s_encoder[:, 0] # ns x 768

