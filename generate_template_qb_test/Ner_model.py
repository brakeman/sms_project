'''模块功能：从上传的 ner格式数据 到 输出一个ner模型'''
'''模块接口：上传的ner格式数据'''

from keras.models import Model, Input, Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

def token2sent(DF):
    sentences = DF.groupby('sent_id').apply(lambda x: ' '.join(x.token.tolist()))
    lables = DF.groupby('sent_id').apply(lambda x: ' '.join(x.lable.tolist()))
    tmp = pd.DataFrame(index=sentences.index, columns=['sms'])
    tmp.sms = sentences.tolist()
    tmp.label = lables.tolist()
    return tmp


def get_X_Y(DF, train=True):
    tmp = token2sent(DF)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>', filters='')
    tokenizer.fit_on_texts(DF.token)
    token_seq = tokenizer.texts_to_sequences(tmp.sms)
    vocab_size = len(tokenizer.word_index) + 1
    label_seq = tmp.label
    maxlen = max([len(i) for i in token_seq])
    ylen = [len(i.split(' ')) for i in label_seq]
    xlen = [len(i) for i in token_seq]
    if ylen != xlen:
        raise Exception('x,y not corresponding well since tokenizer problem')
    tags = DF.lable.tolist()
    tags = list(set(tags))
    n_tags = len(tags)
    print('n_tags:{}'.format(n_tags))
    print('maxlen:{}'.format(maxlen))
    print('vocab_size:{}'.format(vocab_size))
    label2idx = {t: i for i, t in enumerate(tags)}
    print('label2idx:{}'.format(label2idx))


    X_train = pad_sequences(token_seq, padding='post', maxlen=maxlen)
    y_train = [[label2idx[w] for w in s.split(' ')] for s in label_seq]
    y_train = pad_sequences(maxlen=maxlen, sequences=y_train, padding='post', value=label2idx['other'])
    y_train = [to_categorical(i, num_classes=n_tags) for i in y_train]
    print('X_train:{}'.format(X_train.shape))
    print('y_train:{}'.format(y_train[0].shape))
    return X_train, y_train, vocab_size, n_tags, tokenizer, maxlen, label2idx


def lstm_crf(x, y, vocab_size, n_tags, batch_size, epochs):
    output_dim = 30
    hid_size = 50
    dense1 = 50
    seq_len = x.shape[1]
    input_ = Input(shape=(seq_len,))
    model = Embedding(input_dim=vocab_size, output_dim = output_dim,
                     input_length=seq_len, mask_zero=True)(input_)
    model = Bidirectional(LSTM(units=hid_size, return_sequences=True,
                              recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(dense1, activation='relu'))(model)
    crf = CRF(n_tags, learn_mode='marginal')
    out = crf(model) # prob
    model = Model(inputs = input_, outputs = out)
    model.compile(optimizer = 'rmsprop',
                  loss = crf_loss,
                  metrics=[crf.accuracy])
    model.summary()
    history = model.fit(x, np.array(y), batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)
    return model, history


def showRes(new_sms, p, label2idx):
    idx2label = {}
    for k, v in label2idx.items():
        idx2label[v] = k
    p = np.argmax(p, axis =- 1)
    print(p)
    print('{:15}||{:7}'.format('token', 'pred'))
    print(30 * '=')
    for w, pred in zip(new_sms[0], p[0]):
        print('{:25}||{:5}'.format(w, idx2label[pred]))


def predict(model, maxlen, tokenizer, new_sms, show = False, label2idx=None):
    token_seq = tokenizer.texts_to_sequences(new_sms)
    x_te = pad_sequences(token_seq, padding = 'post', maxlen = maxlen)
    p = model.predict(x_te)
    if show:
        showRes(new_sms, p, label2idx)
    return p


def preprocess(text, single=False):
    # if text is a df obj:
    if isinstance(text, pd.DataFrame):
        text_ = text.sms.values.tolist()
        final = []
        for idx, single in enumerate(text_):
            if idx % 50000 == 0:
                print('processing sentence num:{}'.format(idx))
            sentence = ''.join(single)
            sentence = sentence.replace('?', ' ')
            sentence = sentence.replace('!', ' ')
            text2 = sentence.replace(',', ' ')
            text3 = text2.replace('.', '. ')
            text4 = [w.lower() for w in text3.split()]
            text4 = [re.search('^rs\.?', s).group(0) + re.sub('[rs.]+', ' ', s)
                     if re.search('^rs\.?[0-9]', s) else s for s in text4]
            final_sent = ' '.join(text4)
            fin = final_sent.join(' \n').strip()
            final.append(fin)
        text['new_sms'] = final

    # if text is a string obj:
    elif isinstance(text, str):
        sentence = ''.join(text)
        sentence = sentence.replace('?', ' ')
        sentence = sentence.replace('!', ' ')
        text2 = sentence.replace(',', ' ')
        text3 = text2.replace('.', '. ')
        text4 = [w.lower() for w in text3.split()]
        text4 = [re.search('^rs\.?', s).group(0) + re.sub('[rs.]+', ' ', s)
                 if re.search('^rs\.?[0-9]', s) else s for s in text4]
        final_sent = ' '.join(text4)
        text = final_sent.join(' \n').strip()
    elif isinstance(text, list):
        sentence = ' '.join(text).strip()
        sentence = sentence.replace('?', ' ')
        sentence = sentence.replace('!', ' ')
        text2 = sentence.replace(',', ' ')
        text3 = text2.replace('.', '. ')
        text4 = [w.lower() for w in text3.split()]
        text4 = [re.search('^rs\.?', s).group(0) + re.sub('[rs.]+', ' ', s)
                 if re.search('^rs\.?[0-9]', s) else s for s in text4]
        final_sent = ' '.join(text4)
        text = final_sent.join(' \n').strip()
    else:
        raise Exception('text type not allowed')
    return text


