import pandas as pd
import random


def create_fake_label(DF, fake_times):
    DF2 = DF.copy()
    sample_str = 'xX123xX123456xX123xX1237890xX123/*.-2134645623123542346234123'
    for i in range(fake_times):
        DF2['new_tokens_{}'.format(i)] = DF2.apply(lambda x:
                                                 ''.join(random.sample(sample_str, 20))
                                                 if x.lable != 'other' else x.token, axis=1)
    return DF2


def generate_sent_id(sent_id_list, origin_sent_id):
    dic = {}
    for k in origin_sent_id:
        tmp = 'send_id_' + str(sent_id_list[-1])
        sent_id_list.pop()
        dic.setdefault(k, tmp)
    return sent_id_list, dic


def fake2real(df, fake_times):
    lis = []
    sent_id_list = list(range(len(df.sent_id.unique()) * fake_times))
    origin_sent_id = df.sent_id.unique()
    for i in range(fake_times):
        tmp_df = df[['new_tokens_{}'.format(i), 'lable', 'sent_id']]
        tmp_df.columns = ['token', 'lable', 'sent_id']
        sent_id_list, dic = generate_sent_id(sent_id_list, origin_sent_id)
        tmp_df.sent_id = tmp_df.sent_id.map(dic)
        lis.append(tmp_df)
    return pd.concat(lis)
