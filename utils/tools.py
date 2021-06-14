import datetime
import torch
import spacy
from sklearn.metrics import confusion_matrix
from transformers import RobertaModel, RobertaTokenizer


CUDA = torch.cuda.is_available()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', unk_token='<unk>')
nlp = spacy.load("en_core_web_sm")

# Padding function
def padding(sent, pos = False, max_sent_len = 150):
    if pos == False:
        one_list = [1] * max_sent_len
        one_list[0:len(sent)] = sent
        return torch.tensor(one_list, dtype=torch.long)
    else:
        one_list = ["None"] * max_sent_len
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

