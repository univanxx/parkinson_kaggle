import os
import os.path
from tqdm import tqdm
from pathlib import Path
from joblib import dump, load

import pandas as pd
import numpy as np
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV



if os.path.isfile('./files/batches.npy') and os.path.isfile('./files/batches.npy'):
    with open('./files/batches.npy', 'rb') as f:
        batches = np.load(f)
    with open('./files/preds.npy', 'rb') as f:
        preds = np.load(f)
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
    preds = []

    for df_i in tqdm(all_dfs):
        for idx in range(df_i.shape[0] // 64):
            batches.append(df_i.iloc[idx*64:(idx+1)*64, :4].to_numpy())
            preds.append(df_i.iloc[idx*64:(idx+1)*64,  -1].to_numpy())
        batches.append(df_i.iloc[-(df_i.shape[0] % 64):, :4].to_numpy())
        preds.append(df_i.iloc[-(df_i.shape[0] % 64):, -1].to_numpy())

    batches = np.concatenate(batches, axis=0)
    preds = np.concatenate(preds, axis=0)

    Path("./files").mkdir(parents=True, exist_ok=True)
    with open('./files/batches.npy', 'wb') as f:
        np.save(f, batches)
    with open('./files/preds.npy', 'wb') as f:
        np.save(f, preds)

X_train, X_validation, y_train, y_validation = train_test_split(batches, preds, train_size=0.85, random_state=42)

train_dataset = Pool(data=X_train,
                     label=y_train)

eval_dataset = Pool(data=X_validation,
                    label=y_validation)

parameters = {'depth'         : [4,5],
                'learning_rate' : [0.01],
                'iterations'    : [10, 20]
                }

model = CatBoostClassifier(loss_function='MultiClass')

Grid_CBC = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=16, verbose=10)
Grid_CBC.fit(X_train, y_train)

estimator = Grid_CBC.best_estimator_
Path("./models").mkdir(parents=True, exist_ok=True)
dump(estimator, "./models/catboost_model.joblib")
