import sys
sys.path.append('path/to/asr_compare')

from dataset.dataloader_continus import LibriTTSDataset, collate_fn


from torch.utils.data import Dataset, DataLoader, RandomSampler

from model.model_JTFS_continus import IS
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

batchsize=64
trainset = LibriTTSDataset("path/to/wav.scp", "path/to/text")
valset=LibriTTSDataset("path/to/wav.scp", "path/to/text")
# We use a random sampler to shuffle the indices
train_sampler = RandomSampler(trainset)
val_sampler = RandomSampler(valset)

torch.cuda.empty_cache()
state_dict = torch.load("path/to/checkpoint",map_location="cpu")
model.load_state_dict(state_dict,strict=False)
torch.cuda.empty_cache()

train_loader = DataLoader(trainset, batch_size=batchsize, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(valset, batch_size=batchsize, sampler=val_sampler, collate_fn=collate_fn)


checkpoint_callback = ModelCheckpoint(
        dirpath="path/to/checkpoint",
        filename='{epoch:02d}-{asr_loss:.2f}-{val_loss:.2f}',
        save_top_k=3,
        every_n_epochs=1,
        monitor='val_loss',
        mode='min',
        save_last=False
    )

trainer = pl.Trainer(
    max_epochs=100000,
    profiler="simple",
    logger=TensorBoardLogger(name='my_model',save_dir="/path/to/log"),
    accelerator='gpu',
    num_nodes=1,
    devices=8,
    log_every_n_steps=50,
    precision="16-mixed",
    callbacks=[checkpoint_callback],
    #accumulate_grad_batches=2,
    strategy="ddp"
    )

trainer.fit(model, train_loader, val_loader)