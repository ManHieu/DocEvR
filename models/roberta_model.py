import torch
import torch.nn as nn
from transformers import RobertaModel
from utils.constant import CUDA
import os.path as path


class ECIRoberta(nn.Module):
    def __init__(self, num_classes, dataset, mlp_size, roberta_type, finetune, loss=None, sub=True, mul=True):
        super().__init__()
        self.num_classes = num_classes
        self.data_set = dataset
        self.mlp_size = mlp_size
        if path.exists("./pretrained_models/models/{}".format(roberta_type)):
            print("Loading pretrain model from local ......")
            self.roberta = RobertaModel.from_pretrained("./pretrained_models/models/{}".format(roberta_type))
        else:
            print("Loading pretrain model ......")
            self.roberta = RobertaModel.from_pretrained(roberta_type)
        self.sub = sub
        self.mul = mul
        self.finetune = finetune
        if roberta_type == 'roberta-base':
            self.roberta_dim = 768
        if roberta_type == 'roberta-large':
            self.roberta_dim = 1024

        if dataset == "HiEve":
            weights = [993.0/333, 993.0/349, 933.0/128, 933.0/453]
        if dataset == "MATRES":
            weights = [6404.0/3233, 6404.0/2263, 6404.0/232, 6404.0/676,]
        if dataset == "I2B2":
            weights = [3066.0/660, 3066.0/461, 3066.0/1945,]
        if dataset == "TBD":
            weights = [12715.0/2590, 12715.0/2104, 12715.0/836, 12715.0/1060, 12715.0/215, 12715.0/5910,]
        weights = torch.tensor(weights)
        if loss == None:
            self.loss = nn.CrossEntropyLoss(weight=weights)
        else:
            self.loss = loss

        if sub==True and mul==True:
            self.fc1 = nn.Linear(self.roberta_dim*4, self.mlp_size*2)
            self.fc2 = nn.Linear(self.mlp_size*2, num_classes)
        if (sub==True and  mul==False) or (sub==False and mul==True):
            self.fc1 = nn.Linear(self.roberta_dim*3, int(self.mlp_size*1.75))
            self.fc2 = nn.Linear(int(self.mlp_size*1.75), num_classes)
        if not (sub and mul):
            self.fc1 = nn.Linear(self.roberta_dim*2, int(self.mlp_size))
            self.fc2 = nn.Linear(int(self.mlp_size), num_classes)

        # print(self.fc1)
        self.drop_out = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU(0.2, True)
    
    def forward(self, x_sent, y_sent, x_position, y_position, xy):
        batch_size = x_sent.size(0)
        # print(x_sent.size())

        if self.finetune:
            output_x = self.roberta(x_sent)[0]
            output_y = self.roberta(y_sent)[0]
        else:
            with torch.no_grad():
                output_x = self.roberta(x_sent)[0]
                output_y = self.roberta(y_sent)[0]

        output_A = torch.cat([output_x[i, x_position[i], :].unsqueeze(0) for i in range(0, batch_size)])
        output_B = torch.cat([output_y[i, y_position[i], :].unsqueeze(0) for i in range(0, batch_size)])
        # print(output_B.size())
        if self.sub and self.mul:
            sub = torch.sub(output_A, output_B)
            mul = torch.mul(output_A, output_B)
            presentation = torch.cat([output_A, output_B, sub, mul], 1)
        if self.sub==True and self.mul==False:
            sub = torch.sub(output_A, output_B)
            presentation = torch.cat([output_A, output_B, sub], 1)
        if self.sub==False and self.mul==True:
            mul = torch.mul(output_A, output_B)
            presentation = torch.cat([output_A, output_B, mul], 1)
        
        # print(presentation.size())
        presentation = self.drop_out(presentation)
        logits = self.fc2(self.drop_out(self.relu(self.fc1(presentation))))
        loss = self.loss(logits, xy)
        return logits, loss