import sys
sys.path.append('path/to/asr_compare')

from dataset.dataloader_continus import ASRDataset, collate_fn


from torch.utils.data import Dataset, DataLoader, RandomSampler

from model.model_llama2_continus import IS
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
hubert_ckpt_path="path/to/hubert_ckpt"
llama_ckpt_path="path/to/llama_ckpt"



pl.seed_everything(666)

model=IS(hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path,layer=layer)
model=model.to("cuda:3")
batchsize=1
test_set_clean=ASRDataset("path/to/test_clean.scp", "path/to/text_asr_test")
test_set_other=ASRDataset("path/to/test_other.scp", "path/to/text_asr_test")
dataloader_clean = DataLoader(test_set_clean, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)
dataloader_other = DataLoader(test_set_other, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)

torch.cuda.empty_cache()
state_dict = torch.load("path/to/llama2_hubertctcemb_ckpt",map_location="cpu")
model.load_state_dict(state_dict,strict=False)
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