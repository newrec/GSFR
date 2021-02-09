# -*- coding: utf-8 -*-

import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
torch.backends.cudnn.enabled = False

random.seed(1234)

workdir = 'datasets/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Epinions', help='dataset name: Ciao/Epinions/Flixster')

parser.add_argument('--test_prop', default=0.2, help='the proportion of data used for test')
args = parser.parse_args()

df =np.dtype([('uid',np.int32),('iid',np.int32),('rating',np.float32)])


if args.dataset == 'Ciao':
    click_f = np.loadtxt(workdir + 'Ciao/my_rating.txt', dtype=np.int32)
    trust_f = np.loadtxt(workdir + 'Ciao/trust.txt', dtype=np.int32)
elif args.dataset == 'Epinions':
    click_f = np.loadtxt(workdir + 'Epinions/my_rating.txt', dtype=np.int32)
    trust_f = np.loadtxt(workdir + 'Epinions/link.txt', dtype=np.int32)
elif args.dataset == 'Flixster':
    click_f = np.loadtxt(workdir + 'Flixster/my_rating.txt', dtype=df)
    trust_f = np.loadtxt(workdir + 'Flixster/my_trust.txt', dtype=np.int32)
else:
    pass


click_list = []
trust_list = []
trust_temp = []

u_items_list = []
u_users_list = []
u_users_items_list = []
i_users_list = []

pos_u_items_list = []
pos_i_users_list = []

user_count = 0
item_count = 0
rate_count = 0

for s in click_f:
    uid = s[0]
    iid = s[1]

    if args.dataset == 'Ciao':
        label = s[3]
    elif args.dataset == 'Epinions':
        label = s[3]
    elif args.dataset == 'Flixster':
        label = s[2]

    if uid > user_count:
        user_count = uid
    if iid > item_count:
        item_count = iid
    if label > rate_count:
        rate_count = label

    click_list.append([uid, iid, label])

for s in trust_f:
    uid = s[0]
    trust_temp.append(uid)


pos_list = []
pos_trust = []
for i in range(len(click_list)):
    pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))
for i in trust_temp:
    pos_trust.append(i)

pos_list = list(set(pos_list))
pos_trust = list(set(pos_trust))

pos_df = pd.DataFrame(pos_list, columns=['uid', 'iid', 'label'])
filter_pos_list = []
user_in_set, user_out_set = set(), set()

for u in tqdm(range(user_count + 1)):
    hist = pos_df[pos_df['uid'] == u]

    if len(hist) < 5:
        continue

    user_in_set.add(u)

    u_items = hist['iid'].tolist()
    u_ratings = hist['label'].tolist()
    filter_pos_list.extend([(u, iid, rating) for iid, rating in zip(u_items, u_ratings)])

print('user in and out size: ', len(user_in_set), len(user_out_set))
print('data size before and after filtering: ', len(pos_list), len(filter_pos_list))


print('test prop: ', args.test_prop)
pos_list = filter_pos_list

random.shuffle(pos_list)
num_test = int(len(pos_list) * args.test_prop)


test_set = pos_list[:num_test]

valid_set = pos_list[num_test:2 * num_test]

train_set = pos_list[2 * num_test:]


print('Train samples: {}, Valid samples: {}, Test samples: {}, Total samples: {}'.format(len(train_set), len(valid_set),
                                                                                         len(test_set), len(pos_list)))

with open(workdir + args.dataset + '/dataset_filter5.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

pos_df = pd.DataFrame(pos_list, columns=['uid', 'iid', 'label'])
train_df = pd.DataFrame(train_set, columns=['uid', 'iid', 'label'])
valid_df = pd.DataFrame(valid_set, columns=['uid', 'iid', 'label'])
test_df = pd.DataFrame(test_set, columns=['uid', 'iid', 'label'])

click_df = pd.DataFrame(click_list, columns=['uid', 'iid', 'label'])
train_df = train_df.sort_values(axis=0, ascending=True, by='uid')
pos_df = pos_df.sort_values(axis=0, ascending=True, by='uid')


for u in tqdm(range(user_count + 1)):
    hist = train_df[train_df['uid'] == u]
    u_items = hist['iid'].tolist()
    u_ratings = hist['label'].tolist()
    if u_items == []:
        u_items_list.append([(0, 0)])
    else:
        u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])


train_df = train_df.sort_values(axis=0, ascending=True, by='iid')


userful_item_set = set()
for i in tqdm(range(item_count + 1)):
    hist = train_df[train_df['iid'] == i]
    i_users = hist['uid'].tolist()
    i_ratings = hist['label'].tolist()


    if i_ratings == []:
        i_users_list.append([(0, 0)])
    else:
        i_users_list.append([(uid, rating) for uid, rating in zip(i_users, i_ratings)])
        userful_item_set.add(i)

print('item size before and after filtering: ', item_count, len(userful_item_set))

with open(workdir + args.dataset + '/effective_users_and items_filter5.pkl', 'wb') as f:
    pickle.dump(list(user_in_set), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(list(userful_item_set), f, pickle.HIGHEST_PROTOCOL)

count_f_origin, count_f_filter = 0, 0
for s in trust_f:
    uid = s[0]
    fid = s[1]
    count_f_origin += 1
    if uid > user_count or fid > user_count:
        continue
    if uid in user_out_set or fid in user_out_set:
        continue

    trust_list.append([uid, fid])
    count_f_filter += 1

print('u-u relation filter size changes: ', count_f_origin, count_f_filter)
trust_df = pd.DataFrame(trust_list, columns=['uid', 'fid'])
trust_df = trust_df.sort_values(axis=0, ascending=True, by='uid')

count_0, count_1 = 0, 0
for u in tqdm(range(user_count + 1)):
    hist = trust_df[trust_df['uid'] == u]

    u_users = hist['fid'].unique().tolist()
    if u_users == []:
        u_users_list.append([0])
        u_users_items_list.append([[(0, 0)]])
        count_0 += 1
        continue
    else:
        u_users_list.append(u_users)
        uu_items = []
        for uid in u_users:

            uu_items.append(u_items_list[uid])
        u_users_items_list.append(uu_items)
        count_1 += 1

print('trust user with items size: ', count_0, count_1)

with open(workdir + args.dataset + '/list_filter5.pkl', 'wb') as f:
    pickle.dump(u_items_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(u_users_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(u_users_items_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(i_users_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)
