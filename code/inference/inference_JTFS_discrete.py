import sys
sys.path.append('path/to/asr_compare')

from dataset.dataloader_asr_discrete import LibriTTSDataset, collate_fn


from torch.utils.data import Dataset, DataLoader, RandomSampler

from model.model_discrete import IS
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
kmeans=100

folder_path="path/to/folder"
hubert_ckpt_path="path/to/hubert_ckpt"
text_bpe_ckpt_path="path/to/text_bpe_ckpt"

hubert_ckpt_path=os.path.join(folder_path,hubert_ckpt_path)
text_bpe_ckpt_path=os.path.join(folder_path,text_bpe_ckpt_path)
pl.seed_everything(666)

model=IS(text_bpe_ckpt_path=text_bpe_ckpt_path,layer=layer,kmeans=kmeans)

batchsize=1

model=model.to("cuda:0")

torch.cuda.empty_cache()
state_dict = torch.load("path/to/discrete_ckpt",map_location="cpu")
model.load_state_dict(state_dict,strict=False)
torch.cuda.empty_cache()

testcleanwavpth="path/to/test_clean.scp"
textpth="path/to/text_asr_test"
testcleankmeanpth="path/to/features/{kmeans}_deduplicate.pt".format(kmeans=kmeans)
cleanset=LibriTTSDataset(testcleanwavpth, textpth,testcleankmeanpth)

testotherwavpth="path/to/test_other.scp"
textpth="path/to/text_asr_test"
testotherkmeanpth="path/to/features/{kmeans}_deduplicate.pt".format(kmeans=kmeans)
# We use a random sampler to shuffle the indices
otherset=LibriTTSDataset(testotherwavpth, textpth,testotherkmeanpth)

clean_loader = DataLoader(cleanset, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)
other_loader = DataLoader(otherset, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)


import tqdm
fdir="path/to/test_clean"
if not os.path.exists(fdir):
    os.makedirs(fdir)
for i,batch in tqdm.tqdm(enumerate(clean_loader)):
    model.test_asr(batch,fdir)
fdir="path/to/test_other"
if not os.path.exists(fdir):
    os.makedirs(fdir)

for i,batch in tqdm.tqdm(enumerate(other_loader)):
    model.test_asr(batch,fdir)

