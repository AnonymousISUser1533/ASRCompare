import sys
sys.path.append('path/to/asr_compare')

from dataset.dataloader_discrete import LibriTTSDataset, collate_fn


from torch.utils.data import Dataset, DataLoader, RandomSampler

from model.model_llama2_discrete import IS
import torch
from lightning.pytorch import Trainer, LightningDataModule, LightningModule, Callback, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import torch.optim as optim
import math

import os
import argparse
import logging

layer=24
folder_path="path/to/folder"
whisper_ckpt_path="path/to/whisper_ckpt"
llama_ckpt_path="path/to/llama_ckpt"



model=IS(whisper_ckpt_path=whisper_ckpt_path,llama_ckpt_path=llama_ckpt_path,layer=layer)

pl.seed_everything(666)
model=model.to("cuda:0")
batchsize=1

test_setwavpath="path/to/test_clean.scp"
test_settextpath="path/to/text_asr_test"
test_set_kmeanpath="path/to/features/hubert/test/layer24/features_deduplicate_k1500.pt"
clean_test_set=LibriTTSDataset(test_setwavpath, test_settextpath,test_set_kmeanpath)

test_setwavpath="path/to/test_other.scp"
test_settextpath="path/to/text_asr_test"
test_set_kmeanpath="path/to/features/hubert/test/layer24/features_deduplicate_k1500.pt"
other_test_set=LibriTTSDataset(test_setwavpath, test_settextpath,test_set_kmeanpath)

dataloader_clean = DataLoader(clean_test_set, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)
dataloader_other = DataLoader(other_test_set, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)
torch.cuda.empty_cache()
state_dict = torch.load("path/to/llama2_hubertctckemeans_ckpt",map_location="cpu")
model.load_state_dict(state_dict)
torch.cuda.empty_cache()




import tqdm




fdir="path/to/test_clean"
if not os.path.exists(fdir):
    os.makedirs(fdir)
for i,batch in tqdm.tqdm(enumerate(dataloader_clean)):
    model.test_asr(batch,fdir)
fdir="path/to/test_other"
if not os.path.exists(fdir):
    os.makedirs(fdir)

for i,batch in tqdm.tqdm(enumerate(dataloader_other)):
    model.test_asr(batch,fdir)