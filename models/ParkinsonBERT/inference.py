import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from collections import OrderedDict

##############################
# code from transformers.py #
from models.ParkinsonBERT.transformer import BERT4Park
#############################

model = BERT4Park()
state_dict = torch.load('/kaggle/input/bert-predict-parkinson/bert_best_checkpoint.pth', map_location=torch.device('cpu'))

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

max_len = 62

def predict(df, estimator):

    answer_matr = {
        0: [1,0,0],
        1: [0,1,0],
        2: [0,0,1],
        3: [0,0,0]
    }

    # prepare data
    batches = []
    masks = []

    for idx in range(df.shape[0] // max_len):
        batches.append((df.iloc[idx*max_len:(idx+1)*max_len, :].to_numpy())[None,:,:])
        # start of a sequence
        if idx == 0:
            batches[-1] = np.insert(batches[-1], 0, np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
        else:
            batches[-1] = np.insert(batches[-1], 0, 2 * np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
        # end of a sequence
        if idx == df.shape[0] // max_len - 1 and df.shape[0] % max_len == 0:
            batches[-1] = np.insert(batches[-1], batches[-1].shape[1], -np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
        # continue of a sequence
        else:
            batches[-1] = np.insert(batches[-1], batches[-1].shape[1], 2 * np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
        masks.append(np.ones((max_len+2,max_len+2))[None,:,:])
    if df.shape[0] % max_len != 0:
        last_batch = np.zeros((max_len, 3))
        last_batch[:(df.shape[0] % max_len), :] = df.iloc[-(df.shape[0] % max_len):, :].to_numpy()
        batches.append(last_batch[None,:,:])
        # continue of a sequence
        batches[-1] = np.insert(batches[-1], 0, 2 * np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
        # end of a sequence
        batches[-1] = np.insert(batches[-1], df.shape[0] % max_len + 1, -np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
        masks.append(np.ones((max_len+2,max_len+2))[None,:,:])
        masks[-1][:, :, 1 + (df.shape[0] % max_len):-1].fill(0)
        masks[-1][:, 1 + (df.shape[0] % max_len):-1, :].fill(0)

    batches = np.concatenate(batches, axis=0)
    masks = np.concatenate(masks, axis=0)   

    res = estimator(torch.Tensor(batches), torch.Tensor(masks)).argmax(axis=2)[:,1:-1].flatten()
    answer_df = []
    for res_i in res:
        answer_df.append(answer_matr[res_i.item()])

    answer_df = pd.DataFrame(answer_df, columns=['StartHesitation', 'Turn', 'Walking'])
    answer_df = answer_df.reset_index().rename(columns={'index': 'Id'})
    answer_df['Id'] = data[:data.rfind('.')] + '_' + answer_df['Id'].astype(str)
    return answer_df
    
submission = []

for data in os.listdir('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/defog'):
    df = pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/defog/' + data, usecols=['AccV', 'AccML', 'AccAP'])
    submission.append(predict(df, model))
    
for data in os.listdir('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/tdcsfog'):
    df = pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/tdcsfog/' + data, usecols=['AccV', 'AccML', 'AccAP'])
    submission.append(predict(df, model))

submission = []

for data in os.listdir('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/defog'):
    df = pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/defog/' + data, usecols=['AccV', 'AccML', 'AccAP'])
    submission.append(predict(df, model))
    
for data in os.listdir('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/tdcsfog'):
    df = pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/tdcsfog/' + data, usecols=['AccV', 'AccML', 'AccAP'])
    submission.append(predict(df, model))

submission = pd.concat(submission)
submission.to_csv("submission.csv", index=False)