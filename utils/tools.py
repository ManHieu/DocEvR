from collections import defaultdict
import datetime
import re
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
def padding(sent, pos = False, max_sent_len = 150):
    if pos == False:
        one_list = [1] * max_sent_len # pad token id 
        one_list[0:len(sent)] = sent
        return one_list
    else:
        one_list = [0] * max_sent_len # none id 
        one_list[0:len(sent)] = sent
        return one_list

def padding_matrix(matrix, pos=False, max_len=150):
    if pos==False:
        one_matrix = []
        for i in range(len(matrix)):
            one_list = [1] * max_len
            one_list[:len(matrix[i])] = matrix[i]
            one_matrix.append(one_list)
            return one_matrix
    else:
        one_matrix = []
        for i in range(len(matrix)):
            one_list = ['None'] * max_len
            one_list[:len(matrix[i])] = matrix[i]
            one_matrix.append(one_list)
            return one_matrix
        
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

def make_selector_input(target, ctx, sent_id):
    bs = len(target)
    assert len(ctx) == bs and len(sent_id) == bs, 'Each element must be same batch size'
    pad_target = []
    target_len = []
    augm_ctx = []
    ctx_len = []
    max_ns = 0
    for i in range(bs):
        target_len.append(len(target[i]))
        pad_target.append(padding(target[i]))
        augment = [padding(sent) for sent in augment_with_target(target[i], sent_id[i], ctx[i])]
        augm_ctx.append(augment)
        ctx_len.append([len(sent) for sent in ctx[i]])
        max_ns = max(len(augment), max_ns)
    
    pad_target = torch.tensor(pad_target, dtype=torch.long)
    target_len = torch.tensor(target_len, dtype=torch.long)
    augm_ctx = pad_to_max_ns(augm_ctx, max_ns)
    augm_ctx = torch.tensor(augm_ctx, dtype=torch.long)
    return pad_target, target_len, augm_ctx, ctx_len

def make_predictor_input(target, pos_target, position_target, sent_id, ctx, pos_ctx, ctx_id):
    bs = len(target)
    assert len(ctx) == bs and len(sent_id) == bs and len(position_target) == bs, 'Each element must be same batch size'
    augm_target = []
    augm_pos_target = []
    augm_position = []
    for i in range(bs):
        selected_ctx = [step[i] for step in ctx_id]
        augment, position = augment_target(target[i], sent_id[i], position_target[i], ctx[i], selected_ctx)
        pos_augment, pos_position = augment_target(pos_target[i], sent_id[i], position_target[i], pos_ctx[i], selected_ctx)
        assert position == pos_position
        augm_target.append(padding(augment))
        augm_pos_target.append(padding(pos_augment, pos=True))
        augm_position.append(position)
    augm_target = torch.tensor(augm_target, dtype=torch.long)
    augm_pos_target = torch.tensor(augm_pos_target, dtype=torch.long)
    augm_position = torch.tensor(augm_position, dtype=torch.long)    
    return augm_target, augm_pos_target, augm_position

def augment_target(target, sent_id, position_target, ctx, ctx_id):
    augment_target = []
    print(ctx_id)
    print(ctx)
    for id in sorted(ctx_id):
        if id < sent_id:
            augment_target += ctx[id][1:]
            position_target += len(ctx[id][1:])     
        elif id == sent_id:
            augment_target = augment_target + target[1:] + ctx[id]
        else:
            augment_target += ctx[id][1:]
    augment_target = [0] + augment_target
    return augment_target, position_target
            

def pad_to_max_ns(ctx, max_ns):
    sent_len = len(ctx[0][0])
    pad_sent = [1] * sent_len
    for i in range(len(ctx)):
        if len(ctx[i]) < max_ns:
            ctx[i] += [pad_sent] * (max_ns - len(ctx[i]))
    return ctx

def augment_with_target(target_sent, sent_id, ctx):
    augm_ctx = []
    for i in range(len(ctx)):
        ctx_sent = ctx[i]
        if i < sent_id:
            augm = ctx_sent + target_sent[1:]
        else:
            augm = target_sent + ctx_sent[1:]
        augm_ctx.append(augm)
    return augm_ctx

def score(predict, flag, xy, task_weights):
    count = defaultdict(int)
    p_task = defaultdict(list)
    g_task = defaultdict(list)
    
    for i in range(len(predict)):
        p_task[predict[i][0]].append(predict[i][1])
        g_task[flag[i]].append(xy[i])
    
    score = 0.0
    for task in p_task.keys():
        micro_f1 = precision_recall_fscore_support(g_task[task], p_task[task], average='micro')[3]
        score += task_weights[task] * micro_f1
    
    return score


