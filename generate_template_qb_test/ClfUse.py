'''模块功能: 给我一个sms， 能够对其正确分类'''

'''模块接口： showClfResult(res, template_df)'''

from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from .ClfTemplates import get_all_template_path, sample_template_path, sample_template

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
            fin = final_sent.join(' \n')
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
        text = final_sent.join(' \n')

    else:
        raise Exception('text type not allowed')
    return text


def get_cos_similarity(sms, templates):
    '''计算一条新sms 与 每个template 相似度'''

    def cos_sim(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    return [cos_sim(i, sms) for i in templates]


def tfIdfVector(corpus):
    '''corpus is a list of sentences'''
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    x = vectorizer.fit_transform(corpus)
    tfidf = transformer.fit_transform(x)
    return tfidf.toarray()


def showClfResult(res, template_df):
    '''展示一条随机采样的短信, 根据与模版短信相似度 --> 得到分类结果。'''
    template_cleaned = preprocess(template_df)
    template_corpus = template_cleaned.new_sms.tolist()
    template_unique_id = template_cleaned.unique_template_id.tolist()
    template_labels = [i[0] if isinstance(i, list) else i for i in template_cleaned.label]
    new_instance_id = None
    if isinstance(res, pd.DataFrame):
        new_instance_ = res.sample()
        new_instance = new_instance_.sms.tolist()
        tex = new_instance[0]
        new_instance = preprocess(new_instance[0], single=True)
        new_instance_id = new_instance_.unique_tmplate_id
        print('new_instance:{}\n此短信来自于模版:{}\n'.format(tex, new_instance_id.values[0]))

    elif isinstance(res, str):
        new_instance = preprocess(res, single=True)
        print(new_instance)
    else:
        raise Exception('instance type not allowed')
    template_corpus.append(new_instance)

    tfidf = tfIdfVector(template_corpus)
    cos_scores = get_cos_similarity(tfidf[-1], tfidf)[:-1]
    Euc_scores = 'to be done'
    max_score = np.max(cos_scores)
    for i in zip(cos_scores, template_labels, template_unique_id):
        if i[0] == max_score:
            print('元组第三个代表模版ID:{}'.format(i))
            return i[1]


def get_samples(banklist, sampled_template_path):
    '''从 banklist 中对每个银行 的每个模版采样N 个样本'''
    To_train_path = '/data-0/tigergraph/TO_train/'
    # Final_Train = '/data-0/tigergraph/Final_train.txt'
    Train_bank_samples = []
    # with open(Final_Train, 'w+') as ff_train:
    for dispatcher in sampled_template_path:
        dispatch_name  = dispatcher[0].split('/')[-2]
        train_file = To_train_path + dispatch_name + '.txt'
        # 每一个银行
        with open(train_file, 'w+') as ff:
            for template in dispatcher:
                # 每一个模版
                with open(template, "r") as file:
                    tmp1 = file.readlines()
                template_nums = template.split('/')[-1].split('_')[-1]
                template_ids = template.split('/')[-1].split('_')[0]
                to_write = tmp1[-400:]
                for text in to_write:
                    Train_bank_samples.append((text, template_ids, template_nums, dispatch_name))

    res = pd.DataFrame.from_records(Train_bank_samples)
    res.columns = ['sms', 'cluster_id', 'cluster_points', 'bank']
    res = res[res.bank.isin(banklist)]
    res['unique_tmplate_id'] = res['bank'] + '_' + res['cluster_id']
    return res


def labeledDataFromJson(json_path):
    import json
    res=[]
    with open(json_path, 'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
    #         print(dic)
            if dic['annotation'] is not None:
                res.append((dic['content'], dic['annotation']['labels']))
            else:
                res.append((dic['content'], 'error'))

    labeled_df = pd.DataFrame.from_records(res)
    labeled_df.columns = ['sms', 'label']
    return labeled_df


if __name__ == '__main__':

    banklist = ['SBIUPI', 'SBIOTP', 'KOTKBK', 'MYACCT', 'BOBTXN',
                'RBISAY', 'CSHBCK', 'ATMSBI', 'UMOBIL', 'PAYTMB']

    template = '/data-0/tigergraph/template'
    all_templates = get_all_template_path(template_dir = template)
    sampled_template_path = sample_template_path(all_templates)
    sampled_template_df = sample_template(sampled_template_path)

    top_10_train_path = '/home/qibo/项目2_sms/综合/bank-top-10-data/sms_top_10_bank.json'
    labeled_df = labeledDataFromJson(top_10_train_path)
    labeled_df.loc[labeled_df.label == 'error', 'label'] = ['流水转出确认支付密码']

    sampled_template_df['label'] = labeled_df.label.tolist()


    sms = get_samples(banklist, sampled_template_path)
    showClfResult(sms, sampled_template_df)