# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import Counter
import json

# The "nrows=10000" argument reads the first 10000 lines of each file

# df_rand = pd.read_csv("data/log_random_4_22_to_5_08_1k.csv", nrows=10000)
#
# df1 = pd.read_csv("data/log_standard_4_08_to_4_21_1k.csv", nrows=10000)

print("Reading interactions...")
interaction = pd.read_csv("./data/log_standard_4_22_to_5_08_1k.csv",
                          usecols=['user_id', 'video_id', 'is_click', 'tab', 'play_time_ms',
                                   'duration_ms', 'profile_stay_time', 'comment_stay_time'])


print("Reading user features...")
user_features = pd.read_csv("./data/user_features_1k.csv")
print("Processing user features...")
user_active_degree = {'high_active':0, 'full_active':1, 'middle_active':2,
                      'low_active':3, '2_14_day_new':4, '30day_retention':5,
                      'single_low_active':6, 'UNKNOWN':7}
user_features['user_active_degree'].replace(user_active_degree, inplace=True)

follow_user_num_range={'0':0, '(0,10]':1, '(10,50]':2, '(100,150]':3, '(150,250]':4,
                       '(250,500]':5, '(50,100]':6, '500+':7}
user_features['follow_user_num_range'].replace(follow_user_num_range, inplace=True)

fans_user_num_range = {'0':0, '[1,10)':1, '[10,100)':2, '[100,1k)':3,
                       '[1k,5k)':4, '[5k,1w)':5, '[1w,10w)':6, '[10w,100w)':7}
user_features['fans_user_num_range'].replace(fans_user_num_range, inplace=True)

friend_user_num_range = {'0':0, '[1,5)':1, '[5,30)':2, '[30,60)':3,
                         '[60,120)':4, '[120,250)':5, '250+':6}
user_features['friend_user_num_range'].replace(friend_user_num_range, inplace=True)

register_days_range = {'15-30':0, '31-60':1, '61-90':2, '91-180':3,
                       '181-365':4, '366-730':5, '730+':6}
user_features['register_days_range'].replace(register_days_range, inplace=True)

values = {'onehot_feat0': 2, 'onehot_feat1': 7, 'onehot_feat2': 50,
          'onehot_feat3': 1471, 'onehot_feat4': 15, 'onehot_feat5': 34,
          'onehot_feat6': 3, 'onehot_feat7': 118, 'onehot_feat8': 454,
          'onehot_feat9': 7, 'onehot_feat10': 5, 'onehot_feat11': 5,
          'onehot_feat12': 2, 'onehot_feat13': 2, 'onehot_feat14': 2,
          'onehot_feat15': 2, 'onehot_feat16': 2, 'onehot_feat17': 2,}
user_features.fillna(value=values,inplace=True)

print("Reading item features...")
video_features_basic = pd.read_csv("./data/video_features_basic_1k.csv")
# video_features_statistics = pd.read_csv("data/video_features_statistic_1k.csv", nrows=10000)

print("Processing item features...")
video_features_basic.drop(['upload_dt', 'upload_type', 'tag', 'video_duration', 'music_type', 'music_id', 'author_id'], axis=1, inplace=True)

video_type = {'NORMAL':0, 'AD':1, 'UNKNOWN':2}
video_features_basic['video_type'].replace(video_type, inplace=True)


values = {'visible_status':2, 'server_width':0, 'server_height':0}
video_features_basic.fillna(value=values,inplace=True)

print("Merging interactions and user features...")
interaction = pd.merge(interaction, user_features, how='inner', on='user_id', sort=False)

# print("Merging interactions and item features...")
interaction = pd.merge(interaction, video_features_basic, how='inner', on='video_id', sort=False)

interaction.info()
print(interaction.max())

# check non-int value
# for i in interaction['music_type']:
#     if not isinstance(i, int):
#         print(type(i))
#         print(i)

# check nan value
# df2 = interaction[interaction.isnull().values==True]
# print(df2)

# check format
# print(interaction.head(1))

interaction = interaction.astype('int')

import os
os.makedirs("feature_mapping", exist_ok=True)


print("Re-mapping into new id...")
col_names = interaction.columns.values.tolist()
col_names.remove("is_click")
for col in col_names:
    values = interaction[col].unique().tolist()
    ids = np.arange(len(values)).tolist()
    rep = dict(zip(values, ids))
    with open('feature_mapping/'+col+'.json', 'w') as fp:
        json.dump(rep, fp)
    interaction[col] = interaction[col].map(rep)
interaction = interaction.sample(frac=1).reset_index(drop=True)

scenario_summary = dict(Counter(interaction["tab"]))
print("Samples in each scenario: ", scenario_summary)

print("Saving...")
interaction.to_csv("kuairand.csv", index=None)
with open('scenario_summary.json', 'w') as fp:
    json.dump(scenario_summary, fp)

print("Done.")