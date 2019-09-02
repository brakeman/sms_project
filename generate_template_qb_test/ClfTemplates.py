'''模块功能: 从生成好的全部上万个template文件 --> 最终可用的一个template_df'''
'''接口：sample_template()'''
import os
import pandas as pd

def get_all_template_path(template_dir):
    template_lis = []
    for fpathe,dirs,fs in os.walk(template_dir):
        # print(dirs)
        lis2 = []
        for f in fs:
            if f != 'for_label.txt':
                lis2.append(os.path.join(fpathe,f))
        template_lis.append(lis2)
    return template_lis

def sample_template_path(template_lis):
    dispatcher_thresh_lis = []
    for dispatcher in template_lis[1:]:
        idxs = [i.split('/')[-1].split('_')[0] for i in dispatcher]
        nums = [int(i.split('/')[-1].split('_')[1]) for i in dispatcher]
        dic = {}
        ratios = [i / sum(nums) for i in nums]
        for idx, num, ratio in zip(idxs, nums, ratios):
            dic[idx] = [num, ratio]

        lis = sorted(dic.items(), key=lambda d: d[1][1], reverse=True)
        percent = 0
        file_names = []
        for i in lis:
            percent += i[1][1]
            if percent < 0.95 and i[1][1] > 0.002:
                name = i[0] + '_' + str(i[1][0])
                file_names.append(name)
        path = '/'.join(dispatcher[0].split('/')[:-1])
        final_path = [path + '/' + i for i in file_names]
        dispatcher_thresh_lis.append(final_path)
    dispatcher_thresh_lis = [i for i in dispatcher_thresh_lis if i]
    return dispatcher_thresh_lis

def sample_template(dispatcher_thresh_lis, num_samples, banklist=None):
    Train_bank_samples = []
    for dispatcher in dispatcher_thresh_lis:
        dispatch_name  = dispatcher[0].split('/')[-2]
        # 每一个银行
        for template in dispatcher:
            if banklist is not None:
                if template.split('/')[-2] in banklist:
                    # 每一个模版
                    with open(template, "r") as file:
                        tmp1 = file.readlines()
                    template_nums = template.split('/')[-1].split('_')[-1]
                    template_ids = template.split('/')[-1].split('_')[0]
                    to_write = tmp1[-num_samples:]
                    for text in to_write:
                        Train_bank_samples.append((text, template_ids, template_nums, dispatch_name))
            else:
                with open(template, "r") as file:
                    tmp1 = file.readlines()
                template_nums = template.split('/')[-1].split('_')[-1]
                template_ids = template.split('/')[-1].split('_')[0]
                to_write = tmp1[-num_samples:]
                for text in to_write:
                    Train_bank_samples.append((text, template_ids, template_nums, dispatch_name))

    template = pd.DataFrame.from_records(Train_bank_samples)
    template.columns = ['sms', 'cluster_id', 'cluster_points', 'bank']
    # template['template_unique_id'] = template.bank + '_' + template.cluster_id
    # template = template.drop(['cluster_id', 'bank'], axis=1)
    print('total templates among all the banks:{}'.format(template.shape[0]))
    return template


if __name__ == '__main__':
    template = '/data-0/tigergraph/template'
    all_templates_path = get_all_template_path(template_dir = template)
    sampled_template_path = sample_template_path(all_templates_path)
    sampled_template_df = sample_template(sampled_template_path)