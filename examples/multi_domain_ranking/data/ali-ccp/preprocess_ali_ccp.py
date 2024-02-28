import re
import os
import gc
import time
import joblib
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scenario_wise_rec.utils.data import reduce_mem_usage as reduce_mem

random.seed(2022)
np.random.seed(2022)
sample_skeleton_train_path = './sample_skeleton_train.csv'
common_features_train_path = './common_features_train.csv'
sample_skeleton_test_path = './sample_skeleton_test.csv'
common_features_test_path = './common_features_test.csv'
save_path = "./"
write_features_map_path = save_path + 'features_map.pkl'
write_features_path = save_path + 'all_features'
sparse_columns = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '210', '216', '508', '509', '702', '853', '301', '109_14', '110_14', '127_14', '150_14']
dense_columns = ['109_14', '110_14', '127_14', '150_14', '508', '509', '702', '853']
uses_columns = [col for col in sparse_columns] + ['D' + col for col in dense_columns]


def preprocess_data(mode='train'):
    assert mode in ['train', 'test']
    if mode == "test" and not os.path.exists(write_features_map_path):
        print("Error! Please run the train mode first!")
        return
    common_features_path = common_features_train_path if mode == "train" else common_features_test_path
    sample_skeleton_path = sample_skeleton_train_path if mode == "train" else sample_skeleton_test_path

    print(f"Start processing common_features_{mode}")
    common_feat_dict = {}
    with open(common_features_path, 'r') as fr:
        for line in tqdm(fr):
            line_list = line.strip().split(',')
            feat_strs = line_list[2]
            feat_dict = {}
            for fstr in feat_strs.split('\x01'):
                filed, feat_val = fstr.split('\x02')
                feat, val = feat_val.split('\x03')
                if filed in sparse_columns:
                    feat_dict[filed] = feat
                if filed in dense_columns:
                    feat_dict['D' + filed] = val
            common_feat_dict[line_list[0]] = feat_dict

    print('join feats...')
    vocabulary = dict(zip(sparse_columns, [{} for _ in range(len(sparse_columns))]))
    with open(f"{write_features_path}_{mode}.tmp", 'w') as fw:
        fw.write('click,purchase,' + ','.join(uses_columns) + '\n')
        with open(sample_skeleton_path, 'r') as fr:
            for line in tqdm(fr):
                line_list = line.strip().split(',')
                if line_list[1] == '0' and line_list[2] == '1':
                    continue
                feat_strs = line_list[5]
                feat_dict = {}
                for fstr in feat_strs.split('\x01'):
                    filed, feat_val = fstr.split('\x02')
                    feat, val = feat_val.split('\x03')
                    if filed in sparse_columns:
                        feat_dict[filed] = feat
                    if filed in dense_columns:
                        feat_dict['D' + filed] = val
                feat_dict.update(common_feat_dict[line_list[3]])
                feats = line_list[1:3]
                for k in uses_columns:
                    feats.append(feat_dict.get(k, '0'))
                fw.write(','.join(feats) + '\n')
                if mode == "train":
                    for k, v in feat_dict.items():
                        if k in sparse_columns:
                            if v in vocabulary[k]:
                                vocabulary[k][v] += 1
                            else:
                                vocabulary[k][v] = 1

    if mode == "train":
        print('before filter low freq:')
        for k, v in vocabulary.items():
            print(k + ':' + str(len(v)))
        new_vocabulary = dict(zip(sparse_columns, [[] for _ in range(len(sparse_columns))]))
        for k, v in vocabulary.items():
            for k1, v1 in v.items():
                if v1 >= 10:
                    new_vocabulary[k].append(k1)
        vocabulary = new_vocabulary
        print('after filter low freq:')
        for k, v in vocabulary.items():
            print(k + ':' + str(len(v)))
        joblib.dump(vocabulary, write_features_map_path, compress=3)

    print('encode feats...')
    vocabulary = joblib.load(write_features_map_path)
    feat_map = {}
    for feat in sparse_columns:
        feat_map[feat] = dict(zip(vocabulary[feat], range(1, len(vocabulary[feat]) + 1)))
    with open(f"{write_features_path}.{mode}", 'w') as fw:
        fw.write('click,purchase,' + ','.join(uses_columns) + '\n')
        with open(f"{write_features_path}_{mode}.tmp", 'r') as fr:
            fr.readline()  # remove header
            for line in tqdm(fr):
                line_list = line.strip().split(',')
                new_line = line_list[:2]
                for value, feat in zip(line_list[2:], uses_columns):
                    if feat in sparse_columns:
                        new_line.append(str(feat_map[feat].get(value, '0')))
                    else:
                        new_line.append(value)
                fw.write(','.join(new_line) + '\n')


if __name__ == "__main__":
    preprocess_data(mode='train')
    preprocess_data(mode='test')
    train_data = reduce_mem(pd.read_csv(f"{write_features_path}.train"))
    test_data = reduce_mem(pd.read_csv(f"{write_features_path}.test"))
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=2022)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    len_train_data = train_data.shape[0]
    len_val_data = val_data.shape[0]
    len_test_data = test_data.shape[0]
    print(f"train_data : {len_train_data}, val_data: {len_val_data}, test_data:{len_test_data}")
    all_data = pd.concat([train_data, val_data, test_data], axis=0)
    del train_data, val_data, test_data
    gc.collect()
    TARGET = ['click', 'purchase']
    col_name = list(all_data.columns)
    dense_features = ['D' + col for col in dense_columns]
    mms = MinMaxScaler(feature_range=(0, 1))
    all_data[dense_features] = mms.fit_transform(all_data[dense_features])
    all_data = reduce_mem(all_data)
    train_data = all_data[:len_train_data]
    val_data = all_data[len_train_data:-len_test_data]
    test_data = all_data[-len_test_data:]
    print("start save all ")

    train_data.to_csv(save_path + "ali_ccp_train_sample.csv", index=False)
    val_data.reset_index(drop=True).to_csv(save_path + "ali_ccp_val_sample.csv", index=False)
    test_data.reset_index(drop=True).to_csv(save_path + "ali_ccp_test_sample.csv", index=False)
    print("complete")