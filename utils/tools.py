from collections import defaultdict
import datetime
import re
import copy
import numpy as np
from scipy.sparse.construct import rand
import torch
import spacy
from sklearn.metrics import confusion_matrix
from transformers import RobertaModel, RobertaTokenizer
from utils.constant import *
from sklearn.metrics import precision_recall_fscore_support

CUDA = torch.cuda.is_available()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', unk_token='<unk>')
nlp = spacy.load("en_core_web_sm")

# Padding function
def padding(sent, pos = False, max_sent_len = 194):
    if pos == False:
        one_list = [1] * max_sent_len # pad token id
        mask = [0] * max_sent_len
        one_list[0:len(sent)] = sent
        mask[0:len(sent)] = [1] * len(sent)
        return one_list, mask
    else:
        one_list = [0] * max_sent_len # none id 
        one_list[0:len(sent)] = sent
        return one_list
      
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def metric(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    Acc, P, R, F1, _ = CM_metric(CM)
    
    return Acc, P, R, F1, CM

def CM_metric(CM):
    all_ = CM.sum()
    
    Acc = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2] + CM[3][3]) / all_
    P = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0][0:3].sum() + CM[1][0:3].sum() + CM[2][0:3].sum() + CM[3][0:3].sum())
    R = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0].sum() + CM[1].sum() + CM[2].sum())
    F1 = 2 * P * R / (P + R)
    
    return Acc, P, R, F1, CM

def RoBERTa_list(content, token_list = None, token_span_SENT = None):
    encoded = tokenizer.encode(content)
    roberta_subword_to_ID = encoded
    # input_ids = torch.tensor(encoded).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids)
    # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    roberta_subwords = []
    roberta_subwords_no_space = []
    for index, i in enumerate(encoded):
        r_token = tokenizer.decode([i])
        if r_token != " ":
            roberta_subwords.append(r_token)
            if r_token[0] == " ":
                roberta_subwords_no_space.append(r_token[1:])
            else:
                roberta_subwords_no_space.append(r_token)

    roberta_subword_span = tokenized_to_origin_span(content, roberta_subwords_no_space[1:-1]) # w/o <s> and </s>
    roberta_subword_map = []
    if token_span_SENT is not None:
        roberta_subword_map.append(-1) # "<s>"
        for subword in roberta_subword_span:
            roberta_subword_map.append(token_id_lookup(token_span_SENT, subword[0], subword[1]))
        roberta_subword_map.append(-1) # "</s>" 
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, roberta_subword_map
    else:
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, -1

def tokenized_to_origin_span(text, token_list):
    token_span = []
    pointer = 0
    for token in token_list:
        while True:
            if token[0] == text[pointer]:
                start = pointer
                end = start + len(token) - 1
                pointer = end + 1
                break
            else:
                pointer += 1
        token_span.append([start, end])
    return token_span

def sent_id_lookup(my_dict, start_char, end_char = None):
    for sent_dict in my_dict['sentences']:
        if end_char is None:
            if start_char >= sent_dict['sent_start_char'] and start_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']
        else:
            if start_char >= sent_dict['sent_start_char'] and end_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']

def token_id_lookup(token_span_SENT, start_char, end_char):
    for index, token_span in enumerate(token_span_SENT):
        if start_char >= token_span[0] and end_char <= token_span[1]:
            return index

def span_SENT_to_DOC(token_span_SENT, sent_start):
    token_span_DOC = []
    #token_count = 0
    for token_span in token_span_SENT:
        start_char = token_span[0] + sent_start
        end_char = token_span[1] + sent_start
        #assert my_dict["doc_content"][start_char] == sent_dict["tokens"][token_count][0]
        token_span_DOC.append([start_char, end_char])
        #token_count += 1
    return token_span_DOC

def id_lookup(span_SENT, start_char):
    # this function is applicable to RoBERTa subword or token from ltf/spaCy
    # id: start from 0
    token_id = -1
    for token_span in span_SENT:
        token_id += 1
        if token_span[0] <= start_char and token_span[1] >= start_char:
            return token_id
    raise ValueError("Nothing is found. \n span sentence: {} \n start_char: {}".format(span_SENT, start_char))

def pos_to_id(sent_pos):
    id_pos_sent =  [pos_dict.get(pos) if pos_dict.get(pos) != None else 0 
                    for pos in sent_pos]
    return id_pos_sent

# def make_predictor_input(target, pos_target, position_target, sent_id, ctx, pos_ctx, ctx_id, dropout_rate=0.05):
#     bs = len(target)
#     assert len(ctx) == bs and len(sent_id) == bs and len(position_target) == bs, 'Each element must be same batch size'
#     augm_target = []
#     augm_target_mask = []
#     augm_pos_target = []
#     augm_position = []
#     for i in range(bs):
#         if ctx_id != 'warming':
#             if ctx_id != 'all':
#                 selected_ctx = [step[i] for step in ctx_id]
#             else:
#                 selected_ctx = list(range(len(ctx[i])))
#             augment, position = augment_target(target[i], sent_id[i], position_target[i], ctx[i], selected_ctx)
#             pos_augment, pos_position = augment_target(pos_target[i], sent_id[i], position_target[i], pos_ctx[i], selected_ctx)
#             assert position == pos_position
#             augment = word_dropout(augment, position, dropout_rate=dropout_rate)
#             pos_augment = word_dropout(pos_augment, position, is_word=False, dropout_rate=dropout_rate)
#             pad, mask = padding(augment, max_sent_len=400)
#             augm_target.append(pad)
#             augm_target_mask.append(mask)
#             augm_pos_target.append(padding(pos_augment, pos=True, max_sent_len=400))
#             augm_position.append(position)
#         else:
#             augment = word_dropout(target[i], position_target[i], dropout_rate=dropout_rate)
#             pos_augment = word_dropout(pos_target[i], position_target[i], is_word=False, dropout_rate=dropout_rate)
#             pad, mask = padding(augment, max_sent_len=150)
#             augm_target.append(pad)
#             augm_target_mask.append(mask)
#             augm_pos_target.append(padding(pos_augment, pos=True, max_sent_len=150))
#             augm_position.append(position_target[i])

#     augm_target = torch.tensor(augm_target, dtype=torch.long)
#     augm_target_mask = torch.tensor(augm_target_mask, dtype=torch.long)
#     augm_pos_target = torch.tensor(augm_pos_target, dtype=torch.long)
#     augm_position = torch.tensor(augm_position, dtype=torch.long)
#     return augm_target, augm_target_mask, augm_pos_target, augm_position

# def augment_target(target, sent_id, pos, ctx, ctx_id):
#     augment_target = []
#     # print("target: {}".format(target))
#     # print("target pos: {}".format(pos))
#     # print("ctx id: {} ".format(ctx_id))
#     # print("ctx: {}".format(ctx))
#     # print("ctx_len: {}".format(len(ctx)))
#     # print(sent_id)
#     doc = copy.deepcopy(ctx)
#     position_target = pos
#     doc.insert(sent_id, target)
#     new_ctx_id = []
#     for id in ctx_id:
#         if id < sent_id:
#             new_ctx_id.append(id)
#         else:
#             new_ctx_id.append(id + 1)

#     new_ctx_id.append(sent_id)
#     for id in sorted(new_ctx_id):
#         if id < sent_id:
#             augment_target += doc[id][1:]
#             position_target += len(doc[id][1:])
#         else:
#             augment_target += doc[id][1:]
#     augment_target = [0] + augment_target
#     assert augment_target[position_target] == target[pos]
#     # print("augment_target: {}".format(augment_target))
#     # print("position_target: {}".format(position_target))
#     return augment_target, position_target
            
def pad_to_max_ns(ctx_augm_emb):
    max_ns = 0
    ctx_augm_emb_paded = []
    for ctx in ctx_augm_emb:
        # print(ctx.size())
        max_ns  = max(max_ns, ctx.size(0))
    
    pad = torch.zeros((max_ns, 768))
    for ctx in ctx_augm_emb:
        if ctx.size(0) < max_ns:
            pad[:ctx.size(0), :] = ctx
            ctx_augm_emb_paded.append(pad)
        else:
            ctx_augm_emb_paded.append(ctx)
    return ctx_augm_emb_paded

# def augment_ctx(target_sent, target_id, ctx_sent, ctx_id):
#     if ctx_id < target_id:
#         augm = ctx_sent + target_sent[1:]
#     else:
#         augm = target_sent + ctx_sent[1:]
#     return augm

def word_dropout(seq_id, position, is_word=True, dropout_rate=0.05):
    if is_word==True:
        drop_sent = [3 if np.random.rand() < dropout_rate and i != position else seq_id[i] for i in range(len(seq_id))]
    if is_word==False:
        drop_sent = [20 if np.random.rand() < dropout_rate and i != position else seq_id[i] for i in range(len(seq_id))]
    # print(drop_sent)
    return drop_sent

def create_target(x_sent, y_sent, x_position, y_position, x_sent_id, y_sent_id):
    if x_sent_id < y_sent_id:
        sent = x_sent + y_sent[1:]
        x_position_new = x_position
        y_position_new = y_position + len(x_sent) - 1 # remove <s>
    elif x_sent_id == y_sent_id:
        assert x_sent == y_sent
        sent = x_sent
        x_position_new = x_position
        y_position_new = y_position
    else:
        sent = y_sent + x_sent[1:]
        y_position_new = y_position
        x_position_new = x_position + len(y_sent) - 1 # remove <s>
    
    assert x_sent[x_position] == sent[x_position_new]
    assert y_sent[y_position] == sent[y_position_new]
    
    return sent, x_position_new, y_position_new


