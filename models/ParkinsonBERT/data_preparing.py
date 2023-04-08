import numpy as np
import pandas as pd

import os
from tqdm import tqdm
from pathlib import Path


def get_data(max_len=62):
    if os.path.isfile('./files/batches.npy') and os.path.isfile('./files/batches.npy'):
        with open('./files/batches.npy', 'rb') as f:
            batches = np.load(f)
        with open('./files/preds.npy', 'rb') as f:
            preds = np.load(f)
        with open('./files/masks.npy', 'rb') as f:
            masks = np.load(f)
        return batches, masks, preds
    else:
        all_dfs = []
        label2id = {"StartHesitation":0, "Turn":1, "Walking":2, "None":3}

        for file in tqdm(os.listdir("./compete_data/train/defog/")):
            df = pd.read_csv("./compete_data/train/defog/" + file)
            df = df[df.Task == True][['Time', 'AccV', 'AccML', 'AccAP', 'StartHesitation', 'Turn', 'Walking']]
            df['None'] = 1 - df.iloc[:,4:].sum(axis=1)
            df = df.rename(columns=label2id)
            df['y'] = df.iloc[:,4:].idxmax(axis=1)
            all_dfs.append(df)

        for file in tqdm(os.listdir("./compete_data/train/tdcsfog/")):
            df = pd.read_csv("./compete_data/train/tdcsfog/" + file)
            df['None'] = 1 - df.iloc[:,4:].sum(axis=1)
            df = df.rename(columns=label2id)
            df['y'] = df.iloc[:,4:].idxmax(axis=1)
            all_dfs.append(df)

        batches = []
        masks = []
        preds = []

        for df_i in tqdm(all_dfs):
            for idx in range(df_i.shape[0] // max_len):
                batches.append((df_i.iloc[idx*max_len:(idx+1)*max_len, 1:4].to_numpy())[None,:,:])
                # start of a sequence
                if idx == 0:
                    batches[-1] = np.insert(batches[-1], 0, np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
                else:
                    batches[-1] = np.insert(batches[-1], 0, 2 * np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
                # end of a sequence
                if idx == df_i.shape[0] // max_len - 1 and df_i.shape[0] % max_len == 0:
                    batches[-1] = np.insert(batches[-1], batches[-1].shape[1], -np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
                # continue of a sequence
                else:
                    batches[-1] = np.insert(batches[-1], batches[-1].shape[1], 2 * np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
                preds.append((df_i.iloc[idx*max_len:(idx+1)*max_len,  -1].to_numpy())[None,:])
                masks.append(np.ones((max_len+2,max_len+2))[None,:,:])
            if df_i.shape[0] % max_len != 0:
                last_batch = np.zeros((max_len, 3))
                last_batch[:(df_i.shape[0] % max_len), :] = df_i.iloc[-(df_i.shape[0] % max_len):, 1:4].to_numpy()
                batches.append(last_batch[None,:,:])
                # continue of a sequence
                batches[-1] = np.insert(batches[-1], 0, 2 * np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
                # end of a sequence
                batches[-1] = np.insert(batches[-1], df_i.shape[0] % max_len + 1, -np.ones([1,batches[-1].shape[-1]])[None,:,:], axis=1)
                masks.append(np.ones((max_len+2,max_len+2))[None,:,:])
                masks[-1][:, :, 1 + (df_i.shape[0] % max_len):-1].fill(0)
                masks[-1][:, 1 + (df_i.shape[0] % max_len):-1, :].fill(0)

                last_preds = np.zeros((max_len,))
                last_preds[:(df_i.shape[0] % max_len)] = df_i.iloc[-(df_i.shape[0] % max_len):, -1].to_numpy()
                preds.append(last_preds[None,:])

        batches = np.concatenate(batches, axis=0)
        masks = np.concatenate(masks, axis=0)
        preds = np.concatenate(preds, axis=0)

        Path("./files").mkdir(parents=True, exist_ok=True)
        with open('./files/batches.npy', 'wb') as f:
            np.save(f, batches)
        with open('./files/preds.npy', 'wb') as f:
            np.save(f, preds)
        with open('./files/masks.npy', 'wb') as f:
            np.save(f, masks)
        return batches, masks, preds