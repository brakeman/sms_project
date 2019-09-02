'''模块功能: 100多个dispatcher中每个dispatcher: 得到有限模版.'''

'''模块接口'''
import os


def path_from_dir(res_root):
    path_lis = []
    for fpathe, dirs, fs in os.walk(res_root):
        print(fs)
        for f in fs:
            path_lis.append(os.path.join(fpathe, f))

    path_lis = [i for i in path_lis if i != '/data-0/gaoyu/india_sms/BANK_me_origin_20170901/.DS_Store']
    return path_lis


def generate_template_from_sms_files(sms_files):
    for sms_file in sms_files:
        if (sms_file != '/data-0/gaoyu/india_sms/BANK_me_origin_20170901/INDUSB.csv') and (
                sms_file != '/data-0/gaoyu/india_sms/BANK_me_origin_20170901/SBGMBS.csv'):
            template_dir = generate_template(sms_file)
            # get_cluster_stat(redis_client, sms_file, template_dir)
    return
if __name__ == '__main__':
    res_root = '/data-0/gaoyu/india_sms/BANK_me_origin_20170901'
    sms_files = path_from_dir(res_root)
    generate_template_from_sms_files(sms_files)