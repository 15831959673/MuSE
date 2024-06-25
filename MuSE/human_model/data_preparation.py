import itertools
import gensim
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from keras.preprocessing.sequence import pad_sequences
from keras.src.preprocessing.text import Tokenizer

def data_cat(op, filename, x):
    data4 = np.load(f'../datasets/{filename}/{filename}_4mer{x}.npz')
    data5 = np.load(f'../datasets/{filename}/{filename}_5mer{x}.npz')
    data6 = np.load(f'../datasets/{filename}/{filename}_6mer{x}.npz')
    x4 = data4['seq_vec']
    x5 = data5['seq_vec']
    x6 = data6['seq_vec']
    print(type(x4))
    if op == 'concatenate':
        result = np.concatenate((x4, x5, x6), axis=1)
    if op == 'stack':
        result = np.stack((x4, x5, x6), axis=2)
    label = data4['label']

    data4.close()
    data5.close()
    data6.close()
    #
    # print(f'拼接完成，拼接后维度是{result.shape}')
    return torch.tensor(result, dtype=torch.float32), torch.tensor(label, dtype=torch.float32),

def get_tokenizer(k):
    if k == 6:
        # 6mer
        f = ['A', 'C', 'G', 'T']
        c = itertools.product(f, f, f, f, f, f)
        res = []
        for i in c:
            temp = i[0]+i[1]+i[2]+i[3]+i[4]+i[5]
            res.append(temp)
        res = np.array(res)
        NB_WORDS = 4097
        tokenizer = Tokenizer(num_words=NB_WORDS)
        tokenizer.fit_on_texts(res)
        acgt_index = tokenizer.word_index
        acgt_index['null'] = 0
    elif k == 5:
        # 5mer
        f = ['A', 'C', 'G', 'T']
        c = itertools.product(f, f, f, f, f)
        res = []
        for i in c:
            temp = i[0] + i[1] + i[2] + i[3] + i[4]
            res.append(temp)
        res = np.array(res)
        NB_WORDS = 1025
        tokenizer = Tokenizer(num_words=NB_WORDS)
        tokenizer.fit_on_texts(res)
        acgt_index = tokenizer.word_index
        acgt_index['null'] = 0
    elif k == 4:
        # 4mer
        f = ['A', 'C', 'G', 'T']
        c = itertools.product(f, f, f, f)
        res = []
        for i in c:
            temp = i[0] + i[1] + i[2] + i[3]
            res.append(temp)
        res = np.array(res)
        NB_WORDS = 257
        tokenizer = Tokenizer(num_words=NB_WORDS)
        tokenizer.fit_on_texts(res)
        acgt_index = tokenizer.word_index
        acgt_index['null'] = 0
    return tokenizer

def seq(filename, k):
    df = pd.read_csv(f'../datasets/{filename}.csv')
    data = df.values
    word_seq = []
    for i in range(len(data)):
        seq = data[i][0]
        tmp = []
        for i in range(len(seq)-k+1):
            if ('N' in seq[i: i+k]):
                tmp.append('null')
            else:
                tmp.append(seq[i:i+k])
        word_seq.append(' '.join(tmp))
    return word_seq

def word2num(wordseq,tokenizer,MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return numseq

def sentence2num(filename, tokenizer, MAX_LEN, k):
    wordseq=seq(filename, k)
    numseq=word2num(wordseq, tokenizer, MAX_LEN)
    return numseq

def get_data(train_filename, test_filename, k):

    tokenizer = get_tokenizer(k)
    MAX_LEN = 3000
    X_tr = sentence2num(train_filename, tokenizer, MAX_LEN, k)
    X_te = sentence2num(test_filename, tokenizer, MAX_LEN, k)

    return X_tr, X_te

if __name__ == "__main__":
    train_x, test_x = get_data('train_human', 'test_human', 4)
    print(train_x.shape, test_x.shape)