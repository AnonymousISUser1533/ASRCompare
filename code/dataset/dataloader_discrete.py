from torch.utils.data import Dataset, DataLoader, RandomSampler
import math
import numpy as np
import sys
import torch
import random
class LibriTTSDataset(Dataset):
    def __init__(self, mrk_scp, trans, path,num_iter_per_epoch=None):
        self.num_iter_per_epoch = num_iter_per_epoch
        self.mrk_scp = [line.strip() for line in open(mrk_scp).readlines()]
        self.utt_seeker = {}
        self.uttlen = 0
        self.seq_files = {}
        self.id_dict={}
        
        self.wav2idx=torch.load(path)
        self.uttlist = []

        for line in self.mrk_scp:
            # read the mrk file to get the uttid and the seek position and the number of bytes
            skip_first_line = True
            with open(line) as f:
                for utt_line in f:
                    if utt_line=="\n":
                        continue
                    if skip_first_line:
                        # the first line is the total number of utterances
                        self.uttlen += int(utt_line.strip())
                        skip_first_line = False
                    else:
                        # the rest of the lines are the uttid, seek position and the number of bytes
                        utt, seek, num_bytes = utt_line.strip().split()
                        if utt not in self.wav2idx:
                            continue
                        self.uttlist.append(utt)

        self.trans = {}

        with open(trans) as f:
            for line in f:
                uttid, text = line.split()[0], " ".join(line.split()[1:])
                self.trans[uttid] = text


    def __len__(self):
        return len(self.uttlist)

    def __getitem__(self, idx):
        uttid = self.uttlist[idx]
        data = self.wav2idx[uttid]
        return data, self.trans[uttid]
 

import torchaudio
def collate_fn(data_list):
    #datas, texts, lens = [], [], []
    trans_add_prompt=[]
    random_number = 0
    waveforms = []
    targets=[]
    prompts,texts,audios,output_texts,output_audios=[],[],[],[],[]
    #print(random_number)
    for data,text in data_list:
        audios.append(data)
        output_texts.append(text)


    return audios,output_texts

import time
import tqdm

if __name__ == "__main__":
    wavpth="path/to/wav.scp"
    textpth="path/to/text"
    kmeanpth="path/to/features/100_deduplicate.pt"
    dataset_ASR = LibriTTSDataset(wavpth, textpth,kmeanpth)

    # We use a random sampler to shuffle the indices
    sampler = RandomSampler(dataset_ASR)

    dataloader_mrk = DataLoader(
        dataset_ASR,
        shuffle=False,
        num_workers=0,
        batch_size=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    start = time.time()
    for batch_idx, data in tqdm.tqdm(enumerate(dataloader_mrk)): 
        audio,text=data


    print("total number of utts: ", dataset_ASR.uttlen)
    print(time.time() - start)
