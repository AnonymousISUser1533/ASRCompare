import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio


import json
from CODEC.CODEC.models import Generator, Quantizer, Encoder


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class VQVAE(nn.Module):
    def __init__(self, config_path, ckpt_path = None, with_encoder=False):
        super(VQVAE, self).__init__()
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
        
        # load pretrained model
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.generator.load_state_dict(ckpt['generator'])
            self.quantizer.load_state_dict(ckpt['quantizer'])
            if with_encoder:
                self.encoder = Encoder(self.h)
                self.encoder.load_state_dict(ckpt['encoder'])

    def forward(self, x):
        # x is the codebook
        # print('x ', x.shape)
        # assert 1==2
        return self.generator(self.quantizer.embed(x)) # 

    def encode(self, x):
        batch_size = x.size(0)
        if len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        c = self.encoder(x.unsqueeze(1))
        q, loss_q, c = self.quantizer(c)
        c = [code.reshape(batch_size, -1) for code in c]
        # print(torch.stack(c,-1).shape)
        # assert 1==2
        return torch.stack(c, -1) #N, T, 4

class AudioCodec(nn.Module):
    def __init__(self, config_path, model_path):
        super(AudioCodec, self).__init__()
        self.vqvae = VQVAE(config_path, model_path, with_encoder=True)

    def forward(self, wav):
        with torch.no_grad():
            vq_codes = self.vqvae.encode(wav) # 
            syn = self.vqvae(vq_codes)
            return syn

    def vq(self, wav):
        vq_codes = self.vqvae.encode(wav) # 
        return vq_codes

if __name__ == "__main__":
    audiocodec = AudioCodec(
        config_path="/apdcephfs_cq2/share_1603164/user/yaoxunxu/yaoxunxu/work/InstructSpeech/CODEC/CODEC/modelzoo/config_16k_320d.json" ,
        model_path="/apdcephfs_cq2/share_1603164/user/yaoxunxu/yaoxunxu/work/InstructSpeech/CODEC/CODEC/modelzoo/HiFi-Codec-16k-320d" ,
    )  

    wav, sr = torchaudio.load("/apdcephfs_cq2/share_1603164/user/yaoxunxu/yaoxunxu/work/InstructSpeech/CODEC/LJ037-0171.wav")
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    # inference
    audiocodec.vqvae.generator.remove_weight_norm()
    audiocodec.vqvae.encoder.remove_weight_norm()
    audiocodec = audiocodec.eval()
    with torch.no_grad():
        # get codes
        vq = audiocodec.vq(wav)
        print(vq.shape)
        # get wav
        vq=torch.zeros(1,268,4).long()
        wav = audiocodec.vqvae(vq)
    wav = wav.squeeze(1)
    torchaudio.save("codec_16k.wav", wav, 16000)




