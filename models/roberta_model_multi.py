from collections import OrderedDict
from torch.nn.modules.linear import Linear

from torch.utils.data.dataset import Subset
from utils.tools import pos_to_id
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from utils.constant import *
import os.path as path


class ECIRobertaJointTask(nn.Module):
    def __init__(self, mlp_size, roberta_type, datasets,
                finetune, pos_dim=None, loss=None, sub=True, mul=True, fn_activate='relu',
                negative_slope=0.2, drop_rate=0.5, task_weights=None, n_head=3):
        super().__init__()
        
        if path.exists("./pretrained_models/models/{}".format(roberta_type)):
            print("Loading pretrain model from local ......")
            self.roberta = RobertaModel.from_pretrained("./pretrained_models/models/{}".format(roberta_type), output_hidden_states=True)
        else:
            print("Loading pretrain model ......")
            self.roberta = RobertaModel.from_pretrained(roberta_type, output_hidden_states=True)
        if roberta_type == 'roberta-base':
            self.roberta_dim = 768
        if roberta_type == 'roberta-large':
            self.roberta_dim = 1024

        self.sub = sub
        self.mul = mul
        self.finetune = finetune
        if pos_dim != None:
            self.is_pos_emb = True
            pos_size = len(pos_dict.keys())
            self.pos_emb = nn.Embedding(pos_size, pos_dim)
            self.lstm = nn.LSTM(self.roberta_dim+pos_dim, self.roberta_dim//2, num_layers=2, 
                                batch_first=True, bidirectional=True, dropout=drop_rate)
        else:
            self.is_pos_emb = False
            self.lstm = nn.LSTM(self.roberta_dim, self.roberta_dim//2, num_layers=2, 
                                batch_first=True, bidirectional=True, dropout=drop_rate)
        
        self.mlp_size = mlp_size
        self.s_attn = nn.MultiheadAttention(self.roberta_dim, n_head)

        self.drop_out = nn.Dropout(drop_rate)
        if fn_activate=='relu':
            self.relu = nn.LeakyReLU(negative_slope, True)
        elif fn_activate=='tanh':
            self.relu = nn.Tanh()
        elif fn_activate=='relu6':
            self.relu = nn.ReLU6()
        elif fn_activate=='silu':
            self.relu = nn.SiLU()
        elif fn_activate=='hardtanh':
            self.relu = nn.Hardtanh()
        self.max_num_class = 0

        module_dict = {}
        loss_dict = {}
        for dataset in datasets:
            if dataset == "HiEve":
                num_classes = 4
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes
                if sub==True and mul==True:
                    fc1 = nn.Linear(self.roberta_dim*5, int(self.mlp_size*2.5))
                    fc2 = nn.Linear(int(self.mlp_size*2.5), num_classes)
                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.roberta_dim*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)
                if sub==False and mul==False:
                    fc1 = nn.Linear(self.roberta_dim*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)
                
                weights = [993.0/333, 993.0/349, 933.0/128, 933.0/453]
                weights = torch.tensor(weights)
                loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['1'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out), 
                                                ('fc1', fc1), 
                                                ('dropout2', self.drop_out), 
                                                ('relu', self.relu), 
                                                ('fc2',fc2) ]))
                loss_dict['1'] = loss
            
            if dataset == "MATRES":
                num_classes = 4
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes
                if sub==True and mul==True:
                    fc1 = nn.Linear(self.roberta_dim*5, int(self.mlp_size*2.5))
                    fc2 = nn.Linear(int(self.mlp_size*2.5), num_classes)
                if (sub==True and  mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.roberta_dim*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)
                if sub==False and mul==False:
                    fc1 = nn.Linear(self.roberta_dim*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)
                
                weights = [6404.0/3033, 6404.0/2063, 6404.0/232, 6404.0/476,]
                weights = torch.tensor(weights)
                loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['2'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out), 
                                                ('fc1', fc1), 
                                                ('dropout2', self.drop_out), 
                                                ('relu', self.relu), 
                                                ('fc2',fc2) ]))
                loss_dict['2'] = loss
            
            if dataset == "I2B2":
                num_classes = 3
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes
                if sub==True and mul==True:
                    fc1 = nn.Linear(self.roberta_dim*5, int(self.mlp_size*2.5))
                    fc2 = nn.Linear(int(self.mlp_size*2.5), num_classes)
                if (sub==True and  mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.roberta_dim*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)
                if sub==False and mul==False:
                    fc1 = nn.Linear(self.roberta_dim*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)
                
                weights = [3066.0/660, 3066.0/461, 3066.0/1945,]
                weights = torch.tensor(weights)
                loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['3'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out), 
                                                ('fc1', fc1), 
                                                ('dropout2', self.drop_out), 
                                                ('relu', self.relu), 
                                                ('fc2',fc2) ]))
                loss_dict['3'] = loss
            
            if dataset == "TBD":
                num_classes = 6
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes
                if sub==True and mul==True:
                    fc1 = nn.Linear(self.roberta_dim*5, int(self.mlp_size*2.5))
                    fc2 = nn.Linear(int(self.mlp_size*2.5), num_classes)
                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.roberta_dim*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)
                if sub==False and mul==False:
                    fc1 = nn.Linear(self.roberta_dim*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)
                
                weights = [12715.0/2590, 12715.0/2104, 12715.0/836, 12715.0/1060, 12715.0/215, 12715.0/5910,]
                weights = torch.tensor(weights)
                loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['4'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out), 
                                                ('fc1', fc1), 
                                                ('dropout2', self.drop_out), 
                                                ('relu', self.relu), 
                                                ('fc2',fc2) ]))
                loss_dict['4'] = loss
        
        self.module_dict = nn.ModuleDict(module_dict)
        self.loss_dict = nn.ModuleDict(loss_dict)
        self.task_weights = task_weights
        if self.task_weights != None:
            assert len(self.task_weights)==len(datasets), "Length of weight is difference number datasets: {}".format(len(self.task_weights))

    def forward(self, x_sent, y_sent, x_position, y_position, xy, flag, x_sent_pos=None, y_sent_pos=None):
        batch_size = x_sent.size(0)
        # print(x_sent.size())

        if self.finetune:
            output_x = self.roberta(x_sent)[2]
            output_y = self.roberta(y_sent)[2]
        else:
            with torch.no_grad():
                output_x = self.roberta(x_sent)[2]
                output_y = self.roberta(y_sent)[2]
        
        output_x = torch.max(torch.stack(output_x[-4:], dim=0), dim=0)[0]
        output_y = torch.max(torch.stack(output_y[-4:], dim=0), dim=0)[0]
        if x_sent_pos != None and y_sent_pos != None:
            pos_x = self.pos_emb(x_sent_pos)
            pos_y = self.pos_emb(y_sent_pos)
            output_x = torch.cat([output_x, pos_x], dim=2)
            output_y = torch.cat([output_y, pos_y], dim=2)

        output_x = self.drop_out(output_x)
        output_y = self.drop_out(output_y)
        output_x, _ = self.lstm(output_x)
        output_y, _ = self.lstm(output_y)
        # print(output_x.size())
        output_A = torch.cat([output_x[i, x_position[i], :].unsqueeze(0) for i in range(0, batch_size)])
        output_B = torch.cat([output_y[i, y_position[i], :].unsqueeze(0) for i in range(0, batch_size)])

        x, _ = self.s_attn(output_A.unsqueeze(0), output_x.transpose(0,1), output_x.transpose(0,1))
        x = x.squeeze(0)
        y, _ = self.s_attn(output_B.unsqueeze(0), output_y.transpose(0,1), output_y.transpose(0,1))
        y = y.squeeze(0)
        
        if self.sub and self.mul:
            sub = torch.sub(output_A, output_B)
            mul = torch.mul(output_A, output_B)
            sub_s = torch.sub(x, y)
            # mul_s = torch.mul(x, y)
            presentation = torch.cat([output_A, output_B, sub, mul, sub_s], 1)
        if self.sub==True and self.mul==False:
            sub = torch.sub(output_A, output_B)
            sub_s = torch.sub(x, y)
            # mul_s = torch.mul(x, y)
            presentation = torch.cat([output_A, output_B, sub, sub_s], 1)
        if self.sub==False and self.mul==True:
            mul = torch.mul(output_A, output_B)
            sub_s = torch.sub(x, y)
            # mul_s = torch.mul(x, y)
            presentation = torch.cat([output_A, output_B, mul, sub_s], 1)
        if self.sub==False and self.mul==False:
            sub_s = torch.sub(x, y)
            # mul_s = torch.mul(x, y)
            presentation = torch.cat([output_A, output_B, sub_s], 1)
    
        loss = 0.0
        logits = []
        for i in range(0, batch_size):
            typ = str(flag[i].item())
            logit = self.module_dict[typ](presentation[i])
            pad_logit = torch.zeros((1,self.max_num_class))
            pad_logit = pad_logit - 1000
            pad_logit[:, :len(logit)] = logit
            logit = logit.unsqueeze(0)
            target = xy[i].unsqueeze(0)
            if self.task_weights == None:
                loss += self.loss_dict[typ](logit, target)
            else:
                loss += self.task_weights[typ]*self.loss_dict[typ](logit, target)
            
            logits.append(pad_logit)
        return torch.cat(logits, 0), loss