from torch.utils.data import Dataset, DataLoader, RandomSampler
import math
import numpy as np
import sys


class ASRDataset(Dataset):
    def __init__(self, mrk_scp, trans, num_iter_per_epoch=None):
        self.num_iter_per_epoch = num_iter_per_epoch
        self.mrk_scp = [line.strip() for line in open(mrk_scp).readlines()]
        self.seq_scp = [line.strip().replace(".mrk", ".seq") for line in open(mrk_scp).readlines()]
        self.utt_seeker = {}
        self.uttlen = 0
        self.seq_files = {}

        for line in self.mrk_scp:
            # read the mrk file to get the uttid and the seek position and the number of bytes
            skip_first_line = True
            with open(line) as f:
                for utt_line in f:
                    if skip_first_line:
                        # the first line is the total number of utterances
                        self.uttlen += int(utt_line.strip())
                        skip_first_line = False
                    else:
                        # the rest of the lines are the uttid, seek position and the number of bytes
                        utt, seek, num_bytes = utt_line.strip().split()
                        self.utt_seeker[utt] = [line.strip().replace(".mrk", ".seq"), int(seek), int(num_bytes)]

        self.uttlist = list(self.utt_seeker.keys())
        self.trans = {}

        with open(trans) as f:
            for line in f:
                uttid, text = line.split()[0], " ".join(line.split()[1:])
                self.trans[uttid] = text

    def __fetch_one_utt(self, uttid):
        seq_fn, seek, num_bytes = self.utt_seeker[uttid]

        if seq_fn not in self.seq_files:
            # Note, the memory cost directly associated with the seq_files[seq_fn] object is very small, around 4MB per seq file
            self.seq_files[seq_fn] = open(seq_fn, "rb")            

        seq_file = self.seq_files[seq_fn]
        #print(sys.getsizeof(seq_file)/ (1024 * 1024))
        seq_file.seek(seek)            
        num_bytes -= num_bytes % 2
        audio_bytes = seq_file.read(num_bytes)
        audio_np = np.frombuffer(audio_bytes, dtype='int16')

        return audio_np

    def __len__(self):
        return self.uttlen

    def __getitem__(self, idx):
        uttid = self.uttlist[idx]
        data = self.__fetch_one_utt(uttid)
        len_audio = math.ceil(len(data) / 16000)
        return data, self.trans[uttid], len_audio

    def __del__(self):
        for seq_file in self.seq_files.values():
            seq_file.close()    


def collate_fn(data_list):
    datas, texts, lens = [], [], []

    for data, text, len_audio in data_list:
        datas.append(data)
        lens.append(len_audio)
        texts.append(text)

    return datas, texts


if __name__ == "__main__":

    dataset_ASR = ASRDataset("./train.scp", "path/to/text")

    # We use a random sampler to shuffle the indices
    sampler = RandomSampler(dataset_ASR)

    dataloader_mrk = DataLoader(
        dataset_ASR,
        shuffle=False,
        num_workers=4,
        batch_size=32,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
        persistent_workers=True,
        sampler=sampler,
    )

    import time

    start = time.time()
    
    for batch_idx, data in enumerate(dataloader_mrk):          
        print("batch idx: ", batch_idx, data[1], data[2])
        #print("utt: ", i, data[1], data[2])
    print("total number of utts: ", dataset_ASR.uttlen)
    print(time.time() - start)
