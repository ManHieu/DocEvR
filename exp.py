from os import path
import numpy
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import tqdm
from models.predictor_model import ECIRobertaJointTask
from models.selector_model import LSTMSelector, SelectorModel
import time
from utils.tools import *
import torch
import torch.optim as optim


class EXP(object):
    def __init__(self, selector, predictor, num_epoches, num_ctx_select,
                train_dataloader, validate_dataloaders, test_dataloaders, 
                train_short_dataloader, test_short_dataloaders, validate_short_dataloaders,
                s_lr, b_lr, m_lr, decay_rate,  train_lm_epoch, 
                best_path, warmup_proportion=0.1, weight_decay=0.01) -> None:
        super().__init__()
        self.selector = selector
        self.predictor = predictor
        if isinstance(self.selector, SelectorModel):
            self.is_finetune_selector = True
        if isinstance(self.selector, LSTMSelector):
            self.is_finetune_selector = False

        self.num_epoches = num_epoches
        self.num_ctx_select = num_ctx_select

        self.train_dataloader = train_dataloader
        self.test_dataloaders = list(test_dataloaders.values())
        self.validate_dataloaders = list(validate_dataloaders.values())
        self.train_short_dataloader = train_short_dataloader
        self.test_short_dataloaders = list(test_short_dataloaders.values())
        self.validate_short_dataloaders = list(validate_short_dataloaders.values())
        self.datasets = list(test_dataloaders.keys())

        self.s_lr = s_lr
        self.decay_rate = decay_rate
        self.b_lr = b_lr
        self.mlp_lr = m_lr
        self.train_roberta_epoch = train_lm_epoch
        self.warmup_proportion = warmup_proportion

        mlp = ['fc1', 'fc2', 'lstm', 'pos_emb', 's_attn']
        no_decay = ['bias', 'gamma', 'beta']
        group1=['layer.0.','layer.1.','layer.2.','layer.3.']
        group2=['layer.4.','layer.5.','layer.6.','layer.7.']
        group3=['layer.8.','layer.9.','layer.10.','layer.11.']
        group_all = group1 + group2 + group3 
        
        self.b_parameters = [
            {'params': [p for n, p in self.predictor.named_parameters() if not any(nd in n for nd in mlp) and not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.01, 'lr': self.b_lr}, # all params not include bert layers 
            {'params': [p for n, p in self.predictor.named_parameters() if not any(nd in n for nd in mlp) and not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.01, 'lr': self.b_lr*(self.decay_rate**2)}, # param in group1
            {'params': [p for n, p in self.predictor.named_parameters() if not any(nd in n for nd in mlp) and not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.01, 'lr': self.b_lr*(self.decay_rate**1)}, # param in group2
            {'params': [p for n, p in self.predictor.named_parameters() if not any(nd in n for nd in mlp) and not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.01, 'lr': self.b_lr*(self.decay_rate**0)}, # param in group3
            # no_decay
            {'params': [p for n, p in self.predictor.named_parameters() if not any(nd in n for nd in mlp) and any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.00, 'lr': self.b_lr}, # all params not include bert layers 
            {'params': [p for n, p in self.predictor.named_parameters() if not any(nd in n for nd in mlp) and any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.00, 'lr': self.b_lr*(self.decay_rate**2)}, # param in group1
            {'params': [p for n, p in self.predictor.named_parameters() if not any(nd in n for nd in mlp) and any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.00, 'lr': self.b_lr*(self.decay_rate**1)}, # param in group2
            {'params': [p for n, p in self.predictor.named_parameters() if not any(nd in n for nd in mlp) and any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.00, 'lr': self.b_lr*(self.decay_rate**0)}, # param in group3
        ]
        self.mlp_parameters = [
            {'params': [p for n, p in self.predictor.named_parameters() if any(nd in n for nd in mlp) and not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01, 'lr': self.mlp_lr},
            {'params': [p for n, p in self.predictor.named_parameters() if any(nd in n for nd in mlp) and any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.00, 'lr': self.mlp_lr},
            ]
        
        optimizer_parameters = self.b_parameters + self.mlp_parameters 

        self.predictor_optim = optim.AdamW(optimizer_parameters, amsgrad=True, weight_decay=weight_decay)
        self.selector_optim = optim.AdamW(self.predictor.parameters(), lr=self.s_lr, amsgrad=True, weight_decay=weight_decay)

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
        
        lamd = [linear_lr_lambda] * 8
        mlp_lambda = [m_lr_lambda] * 2
        lamd.extend(mlp_lambda)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.predictor_optim, lr_lambda=lamd)
        
        self.best_micro_f1 = [0.0]*len(self.test_dataloaders)
        self.sum_f1 = 0.0
        self.best_matres = 0.0
        self.best_cm = [None]*len(self.test_dataloaders)
        self.best_path_selector = best_path[0]
        self.best_path_predictor = best_path[1]
    
    def task_reward(self, logit, gold):
        logit = torch.softmax(logit, dim=-1)
        reward = []
        for i in range(len(gold)):
            reward.append(logit[i][gold[i]].item())
        reward = numpy.array(reward)
        # print(logit)
        # print(reward)
        return reward - reward.mean()

    def train(self):
        start_time = time.time()
        for i in range(0, self.epochs):
            if i >= self.train_roberta_epoch:
                for group in self.b_parameters:
                    for param in group['params']:
                        param.requires_grad = False

        for i in range(self.num_epoches):
            print("============== Epoch {} / {} ==============".format(i+1, self.num_epoches))
            t0 = time.time()
            self.selector.train()
            self.selector.zero_grad()
            self.selector_loss = 0.0

            self.predictor.train()
            self.predictor.zero_grad()
            self.predictor_loss = 0.0
            if self.train_short_dataloader != None:
                for step, batch in tqdm.tqdm(enumerate(self.train_short_dataloader), desc="Training process for short doc", total=len(self.train_short_dataloader)):
                    x_sent_id, y_sent_id, x_sent, y_sent, x_sent_len, y_sent_len, x_sent_emb, y_sent_emb, x_position, y_position, x_sent_pos, y_sent_pos, \
                    x_ctx, y_ctx, x_ctx_len, y_ctx_len, x_ctx_augm, y_ctx_augm, x_ctx_augm_emb, y_ctx_augm_emb, x_ctx_pos, y_ctx_pos, flag, xy = batch
                    
                    self.predictor_optim.zero_grad()                    
                    p_x_sent, p_x_sent_pos, p_x_position = make_predictor_input(x_sent, x_sent_pos, x_position, x_sent_id, x_ctx, x_ctx_pos, "all")
                    p_y_sent, p_y_sent_pos, p_y_position = make_predictor_input(y_sent, y_sent_pos, y_position, y_sent_id, y_ctx, y_ctx_pos, "all")
                    xy = torch.tensor(xy, dtype=torch.long)
                    flag = torch.tensor(flag, dtype=torch.long)
                    if CUDA:
                        p_x_sent = p_x_sent.cuda() 
                        p_x_sent_pos = p_x_sent_pos.cuda()
                        p_x_position = p_x_position.cuda()
                        p_y_sent = p_y_sent.cuda() 
                        p_y_sent_pos = p_y_sent_pos.cuda()
                        p_y_position = p_y_position.cuda()
                        xy = xy.cuda()
                        flag = flag.cuda()
                    logits, p_loss = self.predictor(p_x_sent, p_y_sent, p_x_position, p_y_position, xy, flag, p_x_sent_pos, p_y_sent_pos)
                    
                    self.predictor_loss += p_loss.item()
                    p_loss.backward()
                    self.predictor_optim.step()
                    self.scheduler.step()
                    

            for step, batch in tqdm.tqdm(enumerate(self.train_dataloader), desc="Training process for long doc", total=len(self.train_dataloader)):
                x_sent_id, y_sent_id, x_sent, y_sent, x_sent_len, y_sent_len, x_sent_emb, y_sent_emb, x_position, y_position, x_sent_pos, y_sent_pos, \
                x_ctx, y_ctx, x_ctx_len, y_ctx_len, x_ctx_augm, y_ctx_augm, x_ctx_augm_emb, y_ctx_augm_emb, x_ctx_pos, y_ctx_pos, flag, xy = batch
                
                self.selector_optim.zero_grad()
                self.predictor_optim.zero_grad()

                if self.is_finetune_selector == False:
                    x_sent_emb = torch.stack(x_sent_emb, dim=0)
                    x_ctx_augm_emb = torch.stack(pad_to_max_ns(x_ctx_augm_emb), dim=0)
                    y_sent_emb = torch.stack(y_sent_emb, dim=0)
                    y_ctx_augm_emb = torch.stack(pad_to_max_ns(y_ctx_augm_emb), dim=0)
                    if CUDA:
                        x_sent_emb = x_sent_emb.cuda()
                        x_ctx_augm_emb = x_ctx_augm_emb.cuda()
                        y_sent_emb = y_sent_emb.cuda()
                        y_ctx_augm_emb = y_ctx_augm_emb.cuda()

                    x_ctx_selected, x_dist, x_log_prob = self.selector(x_sent_emb, x_ctx_augm_emb, x_sent_len, x_ctx_len, self.num_ctx_select)
                    y_ctx_selected, y_dist, y_log_prob = self.selector(y_sent_emb, y_ctx_augm_emb, y_sent_len, y_ctx_len, self.num_ctx_select)
                else:
                    print("This case is not implemented at this time!")
                
                p_x_sent, p_x_sent_pos, p_x_position = make_predictor_input(x_sent, x_sent_pos, x_position, x_sent_id, x_ctx, x_ctx_pos, x_ctx_selected)
                p_y_sent, p_y_sent_pos, p_y_position = make_predictor_input(y_sent, y_sent_pos, y_position, y_sent_id, y_ctx, y_ctx_pos, y_ctx_selected)
                xy = torch.tensor(xy, dtype=torch.long)
                flag = torch.tensor(flag, dtype=torch.long)
                if CUDA:
                    p_x_sent = p_x_sent.cuda() 
                    p_x_sent_pos = p_x_sent_pos.cuda()
                    p_x_position = p_x_position.cuda()
                    p_y_sent = p_y_sent.cuda() 
                    p_y_sent_pos = p_y_sent_pos.cuda()
                    p_y_position = p_y_position.cuda()
                    xy = xy.cuda()
                    flag = flag.cuda()
                logits, p_loss = self.predictor(p_x_sent, p_y_sent, p_x_position, p_y_position, xy, flag, p_x_sent_pos, p_y_sent_pos)
                
                task_reward = self.task_reward(logits, xy)
                s_loss = 0.0
                for i in range(len(task_reward)):
                    s_loss = s_loss - task_reward[i] * (x_log_prob[i] + y_log_prob[i])
                self.selector_loss += s_loss.item()
                self.predictor_loss += p_loss.item()

                s_loss.backward()
                p_loss.backward()
                self.selector_optim.step()
                self.predictor_optim.step()
                self.scheduler.step()
                

            epoch_training_time = format_time(time.time() - t0)
            print("Total training loss: {} - {}".format(self.selector_loss, self.predictor_loss))
            self.evaluate()
        
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-start_time)))
        print("Best micro F1:{}".format(self.best_micro_f1))
        print("Best confusion matrix: ")
        for cm in self.best_cm:
            print(cm)

        return self.best_micro_f1, self.best_cm, self.best_matres

    def evaluate(self, is_test=False):
        F1s = []
        best_cm = []
        sum_f1 = 0.0
        best_f1_mastres = 0.0
        for i in range(0, len(self.test_dataloaders)):
            dataset = self.datasets[i]
            print("-------------------------------{}-------------------------------".format(dataset))
            if is_test:
                dataloader = self.test_dataloaders[i]
                short_dataloader = self.test_short_dataloaders[i]
                self.selector = torch.load(self.best_path_selector)
                self.predictor = torch.load(self.best_path_predictor)
                print("Testset and best model was loaded!")
                print("Running on testset ..........")
            else:
                dataloader = self.validate_dataloaders[i]
                short_dataloader = self.validate_short_dataloaders[i]
                print("Running on validate set ..........")
            
            self.selector.eval()
            self.predictor.eval()
            pred = []
            gold = []
            if short_dataloader != None:
                for step, batch in tqdm.tqdm(enumerate(short_dataloader), desc="Processing for short doc", total=len(short_dataloader)):
                    x_sent_id, y_sent_id, x_sent, y_sent, x_sent_len, y_sent_len, x_sent_emb, y_sent_emb, x_position, y_position, x_sent_pos, y_sent_pos, \
                    x_ctx, y_ctx, x_ctx_len, y_ctx_len, x_ctx_augm, y_ctx_augm, x_ctx_augm_emb, y_ctx_augm_emb, x_ctx_pos, y_ctx_pos, flag, xy = batch
                    
                    self.predictor_optim.zero_grad()                    
                    p_x_sent, p_x_sent_pos, p_x_position = make_predictor_input(x_sent, x_sent_pos, x_position, x_sent_id, x_ctx, x_ctx_pos, "all")
                    p_y_sent, p_y_sent_pos, p_y_position = make_predictor_input(y_sent, y_sent_pos, y_position, y_sent_id, y_ctx, y_ctx_pos, "all")
                    xy = torch.tensor(xy, dtype=torch.long)
                    flag = torch.tensor(flag, dtype=torch.long)
                    if CUDA:
                        p_x_sent = p_x_sent.cuda() 
                        p_x_sent_pos = p_x_sent_pos.cuda()
                        p_x_position = p_x_position.cuda()
                        p_y_sent = p_y_sent.cuda() 
                        p_y_sent_pos = p_y_sent_pos.cuda()
                        p_y_position = p_y_position.cuda()
                        xy = xy.cuda()
                        flag = flag.cuda()
                    logits, p_loss = self.predictor(p_x_sent, p_y_sent, p_x_position, p_y_position, xy, flag, p_x_sent_pos, p_y_sent_pos)
                    labels = xy.cpu().numpy()
                    y_pred = torch.max(logits, 1).indices.cpu().numpy()
                    gold.extend(labels)
                    pred.extend(y_pred)

            for step, batch in tqdm.tqdm(enumerate(dataloader), desc="Processing for long doc", total=len(dataloader)):
                x_sent_id, y_sent_id, x_sent, y_sent, x_sent_len, y_sent_len, x_sent_emb, y_sent_emb, x_position, y_position, x_sent_pos, y_sent_pos, \
                x_ctx, y_ctx, x_ctx_len, y_ctx_len, x_ctx_augm, y_ctx_augm, x_ctx_augm_emb, y_ctx_augm_emb, x_ctx_pos, y_ctx_pos, flag, xy = batch
                
                self.selector_optim.zero_grad()
                self.predictor_optim.zero_grad()

                if self.is_finetune_selector == False:
                    x_sent_emb = torch.stack(x_sent_emb, dim=0)
                    x_ctx_augm_emb = torch.stack(pad_to_max_ns(x_ctx_augm_emb), dim=0)
                    y_sent_emb = torch.stack(y_sent_emb, dim=0)
                    y_ctx_augm_emb = torch.stack(pad_to_max_ns(y_ctx_augm_emb), dim=0)
                    if CUDA:
                        x_sent_emb = x_sent_emb.cuda()
                        x_ctx_augm_emb = x_ctx_augm_emb.cuda()
                        y_sent_emb = y_sent_emb.cuda()
                        y_ctx_augm_emb = y_ctx_augm_emb.cuda()

                    x_ctx_selected, x_dist, x_log_prob = self.selector(x_sent_emb, x_ctx_augm_emb, x_sent_len, x_ctx_len, self.num_ctx_select)
                    y_ctx_selected, y_dist, y_log_prob = self.selector(y_sent_emb, y_ctx_augm_emb, y_sent_len, y_ctx_len, self.num_ctx_select)
                else:
                    print("This case is not implemented at this time!")
                
                print("x_sent_id: ", x_sent_id)
                print("y_sent_id: ", y_sent_id)
                print("x_ctx_selected", x_ctx_selected)
                print("y_ctx_selected", y_ctx_selected)
                p_x_sent, p_x_sent_pos, p_x_position = make_predictor_input(x_sent, x_sent_pos, x_position, x_sent_id, x_ctx, x_ctx_pos, x_ctx_selected)
                p_y_sent, p_y_sent_pos, p_y_position = make_predictor_input(y_sent, y_sent_pos, y_position, y_sent_id, y_ctx, y_ctx_pos, y_ctx_selected)
                xy = torch.tensor(xy, dtype=torch.long)
                flag = torch.tensor(flag, dtype=torch.long)
                if CUDA:
                    p_x_sent = p_x_sent.cuda() 
                    p_x_sent_pos = p_x_sent_pos.cuda()
                    p_x_position = p_x_position.cuda()
                    p_y_sent = p_y_sent.cuda() 
                    p_y_sent_pos = p_y_sent_pos.cuda()
                    p_y_position = p_y_position.cuda()
                    xy = xy.cuda()
                    flag = flag.cuda()
                logits, p_loss = self.predictor(p_x_sent, p_y_sent, p_x_position, p_y_position, xy, flag, p_x_sent_pos, p_y_sent_pos)
                
                labels = xy.cpu().numpy()
                y_pred = torch.max(logits, 1).indices.cpu().numpy()
                gold.extend(labels)
                pred.extend(y_pred)

            P, R, F1 = precision_recall_fscore_support(gold, pred, average='micro')[0:3]
            CM = confusion_matrix(gold, pred)
            print("  P: {0:.3f}".format(P))
            print("  R: {0:.3f}".format(R))
            print("  F1: {0:.3f}".format(F1))
            print("Classification report: \n {}".format(classification_report(gold, pred)))

            if is_test:
                print("Test result:")
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
                torch.save(self.selector, self.best_path_selector)
                torch.save(self.predictor, self.best_path_predictor)
        return F1s