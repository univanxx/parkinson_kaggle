import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from models.ParkinsonBERT.transformer import BERT4Park
from models.ParkinsonBERT.data_preparing import get_data, ParkinsonDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from tqdm.auto import tqdm
import torch.optim as optim
import argparse

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import average_precision_score
import os

import torch.distributed as dist
import torch.multiprocessing as mp


class Trainer():
    def __init__(self, train_dataset, val_dataset, args, gpu):

        torch.cuda.set_device(gpu)

        ############################################################
        rank = args.nr * args.gpus + gpu	                          
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',                                   
            world_size=args.world_size,                              
            rank=rank                                               
        )                                                          
        ############################################################ 
        # if bert_params is not None:
        #     self.model = BERT4Park(**bert_params)
        # else:
        self.model = BERT4Park()

        torch.cuda.set_device(gpu)
        self.model.cuda(gpu)
        self.model = nn.parallel.DistributedDataParallel(self.model.double(), device_ids = [gpu])

        self.batch_size = args.batch_size
        self.len_train = len(train_dataset)
        self.len_val = len(val_dataset)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
    	    num_replicas=args.world_size,
    	    rank=rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=rank,
            shuffle=False
        )

        self.train_loader = DataLoader(train_dataset, pin_memory=True, batch_size=args.batch_size, num_workers=4, sampler=train_sampler)
        self.val_loader = DataLoader(val_dataset, pin_memory=True, batch_size=args.batch_size, num_workers=4, sampler=val_sampler)
        # self.device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
        # self.model = (self.model.double()).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        self.loss = nn.BCELoss().cuda(gpu)

        os.makedirs("./summary", exist_ok=True)
        os.makedirs("./model", exist_ok=True)
        self.writer = SummaryWriter("./summary")
        self.global_step = 1
        self.num_epochs = args.num_epochs

    def train(self):
        best_val = -38
        for epoch in tqdm(range(self.num_epochs)):
            # training
            self.model.train()
            with tqdm(total=np.ceil(self.len_train / self.batch_size).astype(int)) as pbar:
                for batch in self.train_loader:
                    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                    self.optimizer.zero_grad()
                    logits = self.model(batch['value'], batch['mask'])[:,1:-1,:]
                    loss = self.loss(logits, batch['target'])
                    loss.backward()
                    self.optimizer.step()
                    self.writer.add_scalar("Train Loss", loss.item(), global_step=self.global_step)
                    self.writer.add_scalar("Train Macro AP", average_precision_score(batch['target'].reshape(batch['target'].shape[0] * batch['target'].shape[1], 4).cpu().detach(), logits.reshape(logits.shape[0] * logits.shape[1], 4).cpu().detach(), average='macro'), global_step=self.global_step)
                    self.global_step += 1
                    pbar.update(1)
            # validating
            self.model.eval()
            val_loss = []
            with tqdm(total=np.ceil(self.len_val / self.batch_size).astype(int)) as pbar:
                with torch.no_grad():
                    for batch in self.val_loader:
                        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                        logits = self.model(batch['value'], batch['mask'])[:,1:-1,:]
                        val_loss.append(average_precision_score(batch['target'].reshape(batch['target'].shape[0] * batch['target'].shape[1], 4).cpu().detach(), logits.reshape(logits.shape[0] * logits.shape[1], 4).cpu().detach(), average='macro'))
                        pbar.update(1)
            if np.mean(val_loss) > best_val:
                torch.save(self.model.state_dict(), "./model/best_checkpoint.pth")
                best_val = np.mean(val_loss)
            self.writer.add_scalar("Val Macro AP", np.mean(val_loss), global_step=epoch)


def bert_train(gpu, args):

    codebook = {
        1: 'start',
        2: 'cont',
        -1: 'end'
    }
    label2id = {"StartHesitation":0, "Turn":1, "Walking":2, "None":3}
    batches, masks, preds = get_data(max_len=args.max_len)
    X_train, X_validation, y_train, y_validation, masks_train, masks_validation = train_test_split(batches, preds, masks, train_size=0.85, random_state=args.random_state)
    data_train = ParkinsonDataset(X_train, y_train, masks_train)
    data_val = ParkinsonDataset(X_validation, y_validation, masks_validation)      

    train_class = Trainer(data_train, data_val, args, gpu)
    train_class.train()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--num_epochs', default=3, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=256, type=int, 
                        help='batch_size')
    parser.add_argument('--lr', default=3e-3, type=float, 
                        help='learning rate')
    parser.add_argument('--max_len', default=64, type=int, 
                        help='sequence length')
    parser.add_argument('--random_state', default=42, type=int, 
                        help='random state')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = 'localhost'                 #
    os.environ['MASTER_PORT'] = '11238'                     #
    mp.spawn(bert_train, nprocs=args.gpus, args=(args,))    #
    #########################################################



