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

train_path="path/to/train.json"
val_path="path/to/val.json"
import json
with open(train_path, 'r') as f:
    train_json = json.load(f)
with open(val_path, 'r') as f:
    val_json = json.load(f)
trainset=Dataset.from_dict({key: [dic[key] for dic in train_json] for key in train_json[0]})
valset=Dataset.from_dict({key: [dic[key] for dic in val_json] for key in val_json[0]})
from peft import LoraConfig
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM")


from transformers import TrainingArguments
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

model.enable_input_require_grads()
training_arguments = TrainingArguments(
        output_dir="path/to/output", 
        per_device_train_batch_size=1, 
        optim="adamw_torch",
        learning_rate=0.00001, 
        eval_steps=200, 
        save_steps=1000, 
        logging_steps=50, 
                evaluation_strategy="steps",
                group_by_length=False,

        num_train_epochs=100000,

        gradient_accumulation_steps=8, 
        gradient_checkpointing=False, 
        max_grad_norm=0.3,
        bf16=True,
        local_rank=args.local_rank,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        )
from trl import SFTTrainer
trainer = SFTTrainer(
            model=model,  
            train_dataset=trainset,
            eval_dataset=valset,
            dataset_text_field="text",
            peft_config=peft_config,
            max_seq_length=1000,
            tokenizer=tokenizer,
            args=training_arguments,
)

trainer.train()
output_dir="path/to/output"
trainer.model.save_pretrained(output_dir)
