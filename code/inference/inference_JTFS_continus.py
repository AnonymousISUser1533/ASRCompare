import sys
sys.path.append('/apdcephfs_cq2/share_1603164/user/yaoxunxu/yaoxunxu/work/asr_compare')

from dataset.dataloader_asr_continus import LibriTTSDataset, collate_fn


from torch.utils.data import Dataset, DataLoader, RandomSampler

from model.model_continus_whisper import IS
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


folder_path="path/to/folder"
whisper_ckpt_path="path/to/whisper_ckpt"
text_bpe_ckpt_path="path/to/text_bpe_ckpt"

whisper_ckpt_path=os.path.join(folder_path,whisper_ckpt_path)
text_bpe_ckpt_path=os.path.join(folder_path,text_bpe_ckpt_path)
pl.seed_everything(666)

model=IS(whisper_ckpt_path=whisper_ckpt_path,text_bpe_ckpt_path=text_bpe_ckpt_path)

model=model.to("cuda:4")

batchsize=1
test_set_clean=LibriTTSDataset("path/to/test_clean.scp", "path/to/text_asr_test")
test_set_other=LibriTTSDataset("path/to/test_other.scp", "path/to/text_asr_test")
dataloader_clean = DataLoader(test_set_clean, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)
dataloader_other = DataLoader(test_set_other, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)
torch.cuda.empty_cache()
state_dict = torch.load("path/to/continus_whisper_ckpt",map_location="cpu")
model.load_state_dict(state_dict,strict=False)
torch.cuda.empty_cache()

import tqdm
fdir="path/to/continus/test_clean"
if not os.path.exists(fdir):
    os.makedirs(fdir)
for i,batch in tqdm.tqdm(enumerate(dataloader_clean)):
    model.test_asr(batch,fdir)
fdir="path/to/continus/test_other"
if not os.path.exists(fdir):
    os.makedirs(fdir)

for i,batch in tqdm.tqdm(enumerate(dataloader_other)):
    model.test_asr(batch,fdir)