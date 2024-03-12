import sys
sys.path.append('path/to/asr_compare')




from datasets import Dataset
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
import json
import tqdm

layer=24
folder_path="folder_path"
whisper_ckpt_path="path/to/whisper_ckpt"
llama_ckpt_path="path/to/llama_lora_ckpt"

from transformers import AutoModelForCausalLM, AutoTokenizer
# 制定模型名称
model = AutoModelForCausalLM.from_pretrained(
        llama_ckpt_path,
        # load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        )
tokenizer = AutoTokenizer.from_pretrained(llama_ckpt_path)
tokenizer.pad_token_id = 0
model.cuda()
filepath="path/to/dataset"
test_clean_path=filepath+"/test_clean.json"
test_clean_output_path="path/to/output/test_clean"
if not os.path.exists(test_clean_output_path):
    os.makedirs(test_clean_output_path)
target_path=test_clean_output_path+"/target.txt"
predict_path=test_clean_output_path+"/output.txt"
with open(test_clean_path, 'r') as f:
    test_clean_json = json.load(f)
    for i in tqdm.tqdm(range(len(test_clean_json))):
        text=test_clean_json[i]["text"]
        output_start = text.find("Output:") + len("Output: ")
        output_end = text.find("</s>")
        output_content = text[output_start:output_end].strip()

        # 提取前面的内容
        before_output = text[:output_start].strip()
        input_ids = tokenizer.encode(before_output, return_tensors="pt")
        input_ids = input_ids.cuda()
        output = model.generate(input_ids, do_sample=False, max_length=500000)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        output_start = output.find("Output:") + len("Output: ")
        output_prediction = output[output_start:].strip()
        with open(target_path, "a") as f:
            f.write(output_content + "\n")
        with open(predict_path, "a") as f:
            f.write(output_prediction + "\n")

        # # print(before_output)
        # print(output_content)
        # # print(output)
        # print(output_prediction)
        # exit()
test_other_path=filepath+"/test_other.json"
test_other_output_path="path/to/output/test_other"
if not os.path.exists(test_other_output_path):
    os.makedirs(test_other_output_path)
target_path=test_other_output_path+"/target.txt"
predict_path=test_other_output_path+"/output.txt"
with open(test_other_path, 'r') as f:
    test_other_json = json.load(f)
    for i in tqdm.tqdm(range(len(test_other_json))):
        text=test_other_json[i]["text"]
        output_start = text.find("Output:") + len("Output: ")
        output_end = text.find("</s>")
        output_content = text[output_start:output_end].strip()

        # 提取前面的内容
        before_output = text[:output_start].strip()
        input_ids = tokenizer.encode(before_output, return_tensors="pt")
        input_ids = input_ids.cuda()
        output = model.generate(input_ids, do_sample=False, max_length=500000)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        output_start = output.find("Output:") + len("Output: ")
        output_prediction = output[output_start:].strip()
        with open(target_path, "a") as f:
            f.write(output_content + "\n")
        with open(predict_path, "a") as f:
            f.write(output_prediction + "\n")


       
