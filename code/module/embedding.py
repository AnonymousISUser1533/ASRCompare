import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel
from transformers import Wav2Vec2Processor, HubertModel

from typing import List, Union
import logging
import os
import sys
import joblib
import fire
import fairseq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import math
#from einops import rearrange
import re
import numpy as np
from functools import partial
import torch.multiprocessing as mp
import torchaudio
import glob
import tqdm
import argparse
from torchaudio.functional import resample


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x,device):
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x).to(device)
            self.Cnorm = self.Cnorm.to(x).to(device)
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1)
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

class TextEmbedding(nn.Module):
    def __init__(self, embedding_dim,ckpt_path):
        super(TextEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.model = RobertaModel.from_pretrained(ckpt_path)
        #self.Linear=nn.Linear(768,embedding_dim)

    def forward(self, text):
        with torch.no_grad():
            encoded_input = self.tokenizer(text, padding=True, return_tensors='pt')
            encoded_input=encoded_input.to(self.model.device)
            text_mask=encoded_input['attention_mask']
            output = self.model(**encoded_input)
            # output=self.Linear(output[0])
            return text_mask,output[0]
import torchaudio
class AudioEmbedding(nn.Module):
    def __init__(self, embedding_dim,ckpt_path,layer):
        super(AudioEmbedding, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(ckpt_path)
        self.model = HubertModel.from_pretrained(ckpt_path)
        self.layer=layer
        #self.Linear=nn.Linear(1024,embedding_dim)
    def calculate_length(self,x):
        len_ = math.floor((x - 10) / 5 + 1)
        for _ in range(4):
            len_ = math.floor((len_ - 3) / 2 + 1)
        for _ in range(2):
            len_ = math.floor((len_ - 2) / 2 + 1)
        return len_

    def forward(self, audio):
        with torch.no_grad():
            input_values = self.processor(audio, padding=True,return_tensors="pt",sampling_rate=16000)
            input_values=input_values['input_values'].float()
            #print(input_values.shape)
            input_values=input_values.to(self.model.device)
            outputs = self.model(input_values,output_hidden_states=True)['hidden_states']
            output=outputs[self.layer]
            #print(output.shape)
            #output=self.Linear(output)
            #将维度再除以10，10帧结合到一帧
            # for i in audio:
            #     print(len(i))
            len_audio=[self.calculate_length(len(i)) for i in audio]
            len_audio=[math.ceil(i/10) for i in len_audio]

            return len_audio,output

class HubertForCTCEmbedding(nn.Module):
    def __init__(self,embedding_dim,ckpt_path):
        super(HubertForCTCEmbedding, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(ckpt_path)
        self.model = HubertForCTC.from_pretrained(ckpt_path)

    def forward(self, audio):
        with torch.no_grad():
            input_values = self.processor(audio, padding=True,return_tensors="pt",sampling_rate=16000)
            input_values=input_values['input_values']
            logits=self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            return predicted_ids

class TokenEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_model)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def forward(self, x: torch.Tensor):
        X = self.word_embeddings(x)
        X = self.dropout(X)

        return X

class FeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000, fp16=False, sampling_rate=16000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16
        if fp16:
            self.model.half()
        
        self.layer_shift = 0
        self.target_sample_hz = sampling_rate
        

    def read_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.target_sample_hz:
            wav = resample(wav, sr, self.target_sample_hz)
        return wav

    @torch.no_grad()
    def get_feats(self, waveform,device):
        self.model=self.model.to(device)
        x = waveform
        with torch.no_grad():
            if self.fp16:
                x = x.half()
            else:
                x = x.float()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                        source=x_chunk,
                        padding_mask=None,
                        mask=False,
                        output_layer=self.layer + self.layer_shift,
                )
        
                feat.append(feat_chunk)
        if len(feat) == 0:
            return torch.zeros(0, 0)
        return torch.cat(feat, 1).squeeze(0)

class AudioDiscrete(nn.Module):
    def __init__(
        self, 
        ckpt_dir,
        layer=11, 
        max_chunk=1600000, 
        fp16=False, 
        sampling_rate=16000,
        ):

        """
        Args:
            ckpt_dir(str): path to hubert model dir(e.g. hubert_base_ls960.pt)
            layer(int): feat from which layer of hubert models defauly by 9
            max_chunk(int): default by 1600000
            fp16(bool): default by False
            sampling_rate(int): sampling_rate default by 16000
        """
        super().__init__()

        ckpt_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3.pt")
        km_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")

        self.feature_reader = FeatureReader(ckpt_path, layer, max_chunk, fp16, sampling_rate)
        self.apply_kmeans = ApplyKmeans(km_path)

    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i+1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list
    
    def __call__(self,audio,device):
        feat=self.feature_reader.get_feats(audio,device)
        cluster_ids = self.apply_kmeans(feat,device).tolist()
        dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)
        return cluster_ids
    