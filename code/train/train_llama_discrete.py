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

pl.seed_everything(666)

model=IS(whisper_ckpt_path=whisper_ckpt_path,llama_ckpt_path=llama_ckpt_path,layer=layer)
kmeans=1500
batchsize=1
wavpth="path/to/wav.scp"
textpth="path/to/text"
kmeanpth="path/to/features/{kmeans}_deduplicate.pt".format(kmeans=kmeans)
trainset = LibriTTSDataset(wavpth, textpth,kmeanpth)

valwavpth="path/to/dev_wav.scp"
valtextpth="path/to/dev_text"
valkmeanpth="path/to/features/{kmeans}_deduplicate.pt".format(kmeans=kmeans)
valset=LibriTTSDataset(valwavpth, valtextpth,valkmeanpth)

train_sampler = RandomSampler(trainset)
val_sampler = RandomSampler(valset)

train_loader = DataLoader(trainset, batch_size=batchsize, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(valset, batch_size=batchsize, sampler=val_sampler, collate_fn=collate_fn)


checkpoint_callback = ModelCheckpoint(
        dirpath='path/to/checkpoint',
        filename='{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}',
        save_top_k=3,
        every_n_epochs=1,
        monitor='val_loss',
        mode='min',
        save_last=False
    )

trainer = pl.Trainer(
    max_epochs=100000,
    profiler="simple",
    logger=TensorBoardLogger(name='my_model',save_dir='path/to/log'),
    accelerator='gpu',
    num_nodes=1,
    devices=8,
    log_every_n_steps=50,
    precision="bf16-mixed",
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=4,
    strategy="ddp"
    )

trainer.fit(model, train_loader, val_loader)