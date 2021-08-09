import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import numpy
numpy.random.seed(1741)
from os import path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import tqdm
import time
from utils.tools import *
import torch
import torch.optim as optim


class EXP(object):
    def __init__(self, predictor, num_epoches, 
                train_dataloader, validate_dataloaders, test_dataloaders, 
                lr, mlr,
                best_path, warmup_proportion=0.1, word_drop_rate=0.05, weight_decay=0.01) -> None:
        super().__init__()

        self.predictor = predictor

        self.num_epoches = num_epoches

        self.train_dataloader = train_dataloader
        self.test_dataloaders = list(test_dataloaders.values())
        self.validate_dataloaders = list(validate_dataloaders.values())
        
        self.datasets = list(test_dataloaders.keys())

        self.lr = lr
        self.mlr = mlr
        self.warmup_proportion = warmup_proportion
        self.word_drop_rate = word_drop_rate
        mlp = ['fc1', 'fc2', 'lstm', 'pos_emb', 's_attn']
        self.parameters = [
            {'params': [p for n, p in self.predictor.named_parameters() if any(nd in n for nd in mlp)], 'weight_decay_rate': 0.01, 'lr': self.mlr},
            {'params': [p for n, p in self.predictor.named_parameters() if not any(nd in n for nd in mlp)], 'weight_decay_rate': 0.01, 'lr': self.lr},
            ]

        self.optim = optim.AdamW(self.parameters, weight_decay=weight_decay)

        self.train_bert_epoch = 1
        self.num_training_steps = len(self.train_dataloader) * (self.train_bert_epoch)
        self.num_warmup_steps = int(self.warmup_proportion * self.train_bert_epoch)

        def linear_lr_lambda(current_step: int):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            if current_step >= self.num_training_steps:
                return 0
            return max(
                0.0, float(self.num_training_steps - current_step) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            )
        def m_lr_lambda(current_step: int):
            return 0.5 ** int(current_step / (2*len(self.train_dataloader)))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=[m_lr_lambda, linear_lr_lambda])
        
        self.best_micro_f1 = [0.0]*len(self.test_dataloaders)
        self.sum_f1 = 0.0
        self.best_matres = 0.0
        self.best_f1_test = 0.0
        self.best_cm = [None]*len(self.test_dataloaders)
        self.best_path_predictor = best_path
    
    def train(self):
        start_time = time.time()
        for i in range(self.num_epoches):
            print("============== Epoch {} / {} ==============".format(i+1, self.num_epoches))
            if i >= self.train_bert_epoch:
                for param in self.parameters[1]['params']:
                    param.requires_grad = False

            t0 = time.time()
            self.predictor.train()
            self.predictor.zero_grad()
            self.predictor_loss = 0.0

            for step, batch in tqdm.tqdm(enumerate(self.train_dataloader), desc="Training process for long doc", total=len(self.train_dataloader)):
                x, y, x_sent, y_sent, x_sent_id, y_sent_id, x_sent_pos, y_sent_pos, x_position, y_position, x_ev_embs, y_ev_embs, x_kg_ev_emb, \
                y_kg_ev_emb, doc_id, target, target_emb, target_len, ctx, ctx_emb, ctx_ev_embs, num_ev_sents, ctx_ev_kg_embs, ctx_len, ctx_pos, flag, xy = batch
                
                self.optim.zero_grad()                    
                augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position = make_predictor_input(x_sent, y_sent, x_sent_pos, y_sent_pos, x_sent_id, y_sent_id, x_position, y_position, ctx, ctx_pos, 'all', doc_id, dropout_rate=self.word_drop_rate)
                xy = torch.tensor(xy)
                flag = torch.tensor(flag)
                # print(x_kg_ev_emb.size())
                if CUDA:
                    augm_target = augm_target.cuda() 
                    augm_target_mask = augm_target_mask.cuda()
                    augm_pos_target = augm_pos_target.cuda()
                    x_augm_position = x_augm_position.cuda() 
                    y_augm_position = y_augm_position.cuda()
                    xy = xy.cuda()
                    flag = flag.cuda()
                logits, p_loss = self.predictor(augm_target, augm_target_mask, x_augm_position, y_augm_position, xy, flag, augm_pos_target)

                self.predictor_loss += p_loss.item()
                p_loss.backward()
                self.optim.step()
                self.scheduler.step()
            epoch_training_time = format_time(time.time() - t0)
            print("Total training loss: {}".format(self.predictor_loss))
            self.evaluate()
            
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-start_time)))
        print("Best micro F1:{}".format(self.best_micro_f1))
        print("Best confusion matrix: ")
        for cm in self.best_cm:
            print(cm)

        return self.best_micro_f1, self.best_cm, self.best_matres, self.best_f1_test

    def evaluate(self, is_test=False):
        F1s = []
        best_cm = []
        sum_f1 = 0.0
        best_f1_mastres = 0.0
        corpus_labels = {
            "MATRES": 4,
            "TBD": 6,
            "HiEve": 4
        }
        for i in range(0, len(self.test_dataloaders)):
            dataset = self.datasets[i]
            print("-------------------------------{}-------------------------------".format(dataset))
            if is_test:
                dataloader = self.test_dataloaders[i]
                # self.selector = torch.load(self.best_path_selector)
                # self.predictor = torch.load(self.best_path_predictor)
                print("Testset and best model was loaded!")
                print("Running on testset ..........")
            else:
                dataloader = self.validate_dataloaders[i]
                print("Running on validate set ..........")
            
            self.predictor.eval()
            pred = []
            gold = []

            for step, batch in tqdm.tqdm(enumerate(dataloader), desc="Processing for long doc", total=len(dataloader)):
                x, y, x_sent, y_sent, x_sent_id, y_sent_id, x_sent_pos, y_sent_pos, x_position, y_position, x_ev_embs, y_ev_embs, x_kg_ev_emb, \
                y_kg_ev_emb, doc_id, target, target_emb, target_len, ctx, ctx_emb, ctx_ev_embs, num_ev_sents, ctx_ev_kg_embs, ctx_len, ctx_pos, flag, xy = batch
                augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position = make_predictor_input(x_sent, y_sent, x_sent_pos, y_sent_pos, x_sent_id, y_sent_id, x_position, y_position, ctx, ctx_pos, "all", doc_id, dropout_rate=self.word_drop_rate, is_test=True)
                xy = torch.tensor(xy)
                flag = torch.tensor(flag)
                if CUDA:
                    augm_target = augm_target.cuda() 
                    augm_target_mask = augm_target_mask.cuda()
                    augm_pos_target = augm_pos_target.cuda()
                    x_augm_position = x_augm_position.cuda() 
                    y_augm_position = y_augm_position.cuda()
                    xy = xy.cuda()
                    flag = flag.cuda()
                logits, p_loss = self.predictor(augm_target, augm_target_mask, x_augm_position, y_augm_position, xy, flag, augm_pos_target)
                
                labels = xy.cpu().numpy()
                y_pred = torch.max(logits, 1).indices.cpu().numpy()
                gold.extend(labels)
                pred.extend(y_pred)

            CM = confusion_matrix(gold, pred)
            if dataset in  ["MATRES", "TBD"]:
                # no eval in vague
                num_label = corpus_labels[dataset]
                true = sum([CM[i, i] for i in range(num_label-1)])
                sum_pred = sum([CM[i, 0:(num_label-1)].sum() for i in range(num_label)])
                sum_gold = sum([CM[i].sum() for i in range(num_label-1)])
                P = true / sum_pred
                R = true / sum_gold
                F1 = 2 * P * R / (P + R)
                print("  P: {0:.3f}".format(P))
                print("  R: {0:.3f}".format(R))
                print("  F1: {0:.3f}".format(F1))
                print("  Confusion Matrix")
                print(CM)
                print("Classification report: \n {}".format(classification_report(gold, pred)))          
            elif dataset == "HiEve":
                num_label = corpus_labels[dataset]
                true = sum([CM[i, i] for i in range(2)])
                sum_pred = sum([CM[i, 0:2].sum() for i in range(num_label)])
                sum_gold = sum([CM[i].sum() for i in range(2)])
                P = true / sum_pred
                R = true / sum_gold
                F1 = 2 * P * R / (P + R)
                print("  P: {0:.3f}".format(P))
                print("  R: {0:.3f}".format(R))
                print("  F1: {0:.3f}".format(F1))
                print("  Confusion Matrix")
                print(CM)
                print("Classification report HiEve: \n {}".format(classification_report(gold, pred)))
            else:
                P, R, F1 = precision_recall_fscore_support(gold, pred, average='micro')[0:3]
                print("  P: {0:.3f}".format(P))
                print("  R: {0:.3f}".format(R))
                print("  F1: {0:.3f}".format(F1))
                print("  Confusion Matrix")
                print(CM)
                print("Classification report: \n {}".format(classification_report(gold, pred)))
            
            sum_f1 += F1
            best_cm.append(CM)
            F1s.append(F1)
            if dataset=="MATRES": 
                best_f1_mastres=F1

        if is_test==False:
            if sum_f1 > self.sum_f1 or path.exists(self.best_path_selector) == False:
                self.sum_f1 = sum_f1
                self.best_cm = best_cm
                self.best_micro_f1 = F1s 
            if best_f1_mastres > self.best_matres:
                self.best_matres = best_f1_mastres
        else:
            if self.best_f1_test < F1:
                self.best_f1_test = F1
                torch.save(self.selector, self.best_path_selector)
                torch.save(self.predictor, self.best_path_predictor)
        return F1s
        