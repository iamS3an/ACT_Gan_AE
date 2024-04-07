import pandas as pd
import numpy as np
import os, random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess(save_path):
    # 讀取資料集
    train_df = pd.read_csv("data/0_ori/UNSW_NB15_training-set.csv", index_col=0, low_memory=False).reset_index(drop=True)
    test_df = pd.read_csv("data/0_ori/UNSW_NB15_testing-set.csv", index_col=0, low_memory=False).reset_index(drop=True)
    
    # 特徵選擇
    # feature_select = [
    #     'dur', 'service', 'spkts', 'dpkts', 'dbytes', 'sttl', 'sload', 'dload', 'sloss', 'dloss', 'stcpb', 'dtcpb',
    #     'synack', 'smeansz', 'dmeansz', 'ct_srv_src', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    #     'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
    # ]
    # feature_select.append('label')
    # dataframe = dataframe[feature_select]

    # 二分類，attack_cat刪除
    train_df.drop(columns=['attack_cat'], inplace=True)
    test_df.drop(columns=['attack_cat'], inplace=True)

    # label encoding
    for col in train_df.columns:
        if train_df[col].dtype != 'object':
            train_df[col].apply(pd.to_numeric, errors='coerce').astype(np.float64)
            test_df[col].apply(pd.to_numeric, errors='coerce').astype(np.float64)
        else:
            print(col, train_df[col].dtype)
            # 名词列: 删除字符串前后空白，并将其改成小写
            train_df[col] = train_df[col].map(lambda x: x.strip().lower())
            test_df[col] = test_df[col].map(lambda x: x.strip().lower())
            # label encoding
            le = LabelEncoder()
            combine_df = pd.concat([train_df[col], test_df[col]], axis=0).reset_index(drop=True)
            le.fit(combine_df)
            train_df[col] = le.transform(train_df[col])
            test_df[col] = le.transform(test_df[col])

    # 確認資料型態都是float
    print(train_df.info())
    print(test_df.info())

    # 確認label分布
    print(train_df['label'].value_counts())
    print(test_df['label'].value_counts())

    # 確認detail label分布
    # print(train_df['attack_cat'].value_counts())
    # print(test_df['attack_cat'].value_counts())

    # 儲存label，以免label被標準化
    train_label = train_df['label']
    train_df.drop(columns=['label'], inplace=True)
    test_label = test_df['label']
    test_df.drop(columns=['label'], inplace=True)

    # 儲存training_set每个列的最小值和最大值
    min_values = train_df.min()
    max_values = train_df.max()

    # Min-Max 标准化到 0~1 范围(test可能超過)
    train_normalized_df = (train_df - min_values) / (max_values - min_values)
    test_normalized_df = (test_df - min_values) / (max_values - min_values)
    del train_df, test_df

    # 將label放回
    train_normalized_df['label'] = train_label
    test_normalized_df['label'] = test_label

    # 儲存
    os.makedirs(save_path, exist_ok=True)
    train_normalized_df.to_csv(os.path.join(save_path, 'train.csv'), index=None)
    test_normalized_df.to_csv(os.path.join(save_path, 'test.csv'), index=None)

if __name__ == '__main__':
    save_path = os.path.join('data', '1_preprocess')
    preprocess(save_path)