import lightning.pytorch as pl
from typing import Tuple, Union
from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModelForCausalLM
from module.modeling_llama_huggingface import LlamaForCausalLM
from module.transformer import (
    AdaptiveLayerNorm,
    LayerNorm,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from module.layers import ConvNorm, LinearNorm
from module.embedding import TextEmbedding,AudioEmbedding,TokenEmbedding
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from module.utils import make_pad_mask_number, Transpose

from transformers import HubertModel, Wav2Vec2Processor

from torch.nn import CrossEntropyLoss
import os

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, tokenizer_id,ignore_index=-100, eos_penalty=10.0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eos_penalty = eos_penalty
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.tokenizer_id=tokenizer_id

    def forward(self, logits, labels):
        # 计算基础的交叉熵损失
        # print(logits.shape)
        # print(labels.shape)
        base_loss = self.loss_fct(logits, labels)

        # 找到 logits 中预测的 token
        predicted_tokens = torch.argmax(logits, dim=-1)

        # 找到 logits 中与 labels 的 eos 位置不匹配的地方
        # print(self.tokenizer_id)
        # print((predicted_tokens == self.tokenizer_id) )
        # print(labels == self.tokenizer_id)
        eos_mismatch_mask = (predicted_tokens != labels) & ((predicted_tokens == self.tokenizer_id) | (labels == self.tokenizer_id))

        # 计算 eos 位置不匹配的损失
        mismatch_loss = eos_mismatch_mask.sum().float() * self.eos_penalty

        # 计算总损失
        total_loss = base_loss + mismatch_loss
        # print("baseloss:",base_loss)
        # print("mismatch_loss:",mismatch_loss)

        return total_loss

class IS(pl.LightningModule):
    def __init__(self,
        decoder_cls: Union[
            nn.TransformerDecoder, nn.TransformerEncoder
        ] = nn.TransformerDecoder,
        decoder_layer_cls: Union[
            TransformerDecoderLayer, TransformerEncoderLayer
        ] = TransformerDecoderLayer,
        d_model: int = 1024,
        nhead: int = 8,
        num_layers: int = 10,
        hubert_ckpt_path: str = None,
        llama_ckpt_path: str = None,
        layer: int = 24,
        ):
        super(IS, self).__init__()
        self.audio_model = HubertModel.from_pretrained(hubert_ckpt_path)
        self.processor = Wav2Vec2Processor.from_pretrained(hubert_ckpt_path)
        self.text_tokenizer = AutoTokenizer.from_pretrained(llama_ckpt_path)

        self.llama=LlamaForCausalLM.from_pretrained(llama_ckpt_path,torch_dtype=torch.bfloat16)
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.unk_token
        self.bos_emb=self.llama.get_input_embeddings()(torch.tensor(self.text_tokenizer.bos_token_id))
        self.eos_emb=self.llama.get_input_embeddings()(torch.tensor(self.text_tokenizer.eos_token_id))
        for param in self.parameters():
            param.requires_grad = False
        
        self.audio_embedding_last_Linear = nn.Sequential(
            nn.Linear(10240, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096)
        )

        #【[GPT_Begin]8  [Text_Begin]text[Text_End] [Audio_Begin]Audio[Audio_End] [GPT_End]9】NULL
        #7个token，最后一个表示其他无关的tts表征

        
        
        self.accuracy5 = MulticlassAccuracy(
            num_classes=len(self.text_tokenizer),
            top_k=5,
            average="micro",
            multidim_average="global",
            )
        self.accuracy1 = MulticlassAccuracy(
            num_classes=len(self.text_tokenizer),
            top_k=1,
            average="micro",
            multidim_average="global",
            )
        self.accuracy10 = MulticlassAccuracy(
            num_classes=len(self.text_tokenizer),
            top_k=10,
            average="micro",
            multidim_average="global",
            )
        self.wrong_number=0
        self.loss_fn=CustomCrossEntropyLoss(self.text_tokenizer.eos_token_id)

    def forward(self,inputs):
        audio,output_text=inputs
        batchsize=len(audio)
        #print(audio[0].shape[0])
        audio_len=[audio[i].shape[0]//320-1 for i in range(batchsize)]
        audio_len=[(i+9)//10 for i in audio_len]
        # # print(audio.shape)
        # for i in range(batchsize):
        #     audio[i]=audio[i][:audio_len[i]*10]

        #print(audio_len)

        with torch.no_grad():
            input_values = self.processor(audio,  return_tensors="pt", sampling_rate=16000, padding=True).input_values.to(self.device)
            input_values=input_values.half()
            outputs=self.audio_model(input_values,output_hidden_states=True)['hidden_states']
            outputs=outputs[-1]
            # print(outputs)
            print(outputs.shape)
            time_len=outputs.shape[1]
            padded_time_len=(time_len+9)//10*10
            # print(padded_time_len)
            outputs=F.pad(outputs,(0,0,0,padded_time_len-time_len))
            # print(outputs.shape)
            outputs=outputs.view(batchsize,-1,10240)

        # audio_inputs=padded_audio_input_values.contiguous().view(batchsize,padded_time_len//10,-1)
        audio_inputs=outputs.contiguous()
        # print(audio_inputs.shape)
        torch.cuda.empty_cache()
        audio_inputs=self.audio_embedding_last_Linear(audio_inputs)
        #print(audio_len)
        audio_lengths = torch.tensor(audio_len)

        x=[]
        len_x1=[]
        for i in range(batchsize):

            audio_input=audio_inputs[i,:audio_lengths[i],:]
            #print(text_input_pre.shape,audio_input.shape,text_input_post.shape)
            bos_emb=self.bos_emb.unsqueeze(0).to(self.device).detach()
            # eos_emb=self.eos_emb.unsqueeze(0).to(self.device).detach()
            input_x=torch.cat((bos_emb,audio_input),dim=0)
            x.append(input_x)
            #print(text_lengths[i],audio_lengths[i])
            len_x1.append(input_x.shape[0])
        #x为一个list，保存的是每个batch的input，每一个元素代表一个text和一个audio拼接的input
        #len_x为一个list，保存的是每个batch的input的长度
        padded_x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
        x = torch.stack(list(padded_x), dim=0)
        
        texts=[text+self.text_tokenizer.eos_token for text in output_text]
        texts=self.text_tokenizer(texts,return_tensors="pt",padding="longest",truncation=True,add_special_tokens=True).to(self.device)

        targets=texts["input_ids"].masked_fill(
            texts.input_ids == self.text_tokenizer.pad_token_id, -100
        )
        # print(targets)
        with torch.no_grad():
            texts_embes=self.llama.get_input_embeddings()(texts["input_ids"])

        inputs_embeds=torch.cat((x,texts_embes),dim=1)
        # print(x.shape,texts_embes.shape,inputs_embeds.shape)

        attns_text=texts.attention_mask
        attns_audio=make_pad_mask_number(audio_lengths+1).to(x.device)
        #加2是因为bos和eos
        attns=torch.cat((attns_audio,attns_text),dim=1)
        # print(attns_audio.shape,attns_text.shape,attns.shape)
        # print(inputs_embeds.shape,attns.shape,targets.shape)

        outputs=self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attns,
            return_dict=True)
        # print(outputs.keys())
        hidden=outputs[0]
        len_target=targets.shape[1]
        shift_logits = hidden[..., -len_target:-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        loss_fct = self.loss_fn
        shift_logits = shift_logits.view(-1, self.text_tokenizer.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels
        loss = loss_fct(shift_logits, shift_labels)
        # print(loss)






           
        return loss
    def training_step(self,batch,batch_idx):
        loss=self(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,batch_size=len(batch[0]),sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
        return optimizer
    def validation_step(self,batch,batch_idx):
        loss=self(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True,batch_size=len(batch[0]),sync_dist=True)
        return loss

    

    def test_asr(self,inputs,filedir):
        y=self.inference(inputs)
        y_pre=y[0]
        print("y_pre:")
        print(y_pre)
        print("target:")
        print(inputs[1])
        try:
            # 提取两个位置之间的数（不包括这两个数）
            target_dir=os.path.join(filedir,"target.txt")
            output_dir=os.path.join(filedir,"output.txt")
            if not os.path.exists(target_dir):
                with open(target_dir,"w") as f:
                    pass
            if not os.path.exists(output_dir):
                with open(output_dir,"w") as f:
                    pass
            with open(target_dir,"a") as f:
                print(inputs[1][0])
                print(1)
                f.write(inputs[1][0]+"\n")
            with open(output_dir,"a") as f1:
                print(2)
                f1.write(y_pre+"\n")
            
        except:
            self.wrong_number+=1
    def topk_sampling(self,logits, top_k=10, top_p=1.0, temperature=1.0):
        # temperature: (`optional`) float
        #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        # top_k: (`optional`) int
        #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
        # top_p: (`optional`) float
        #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

        # Temperature (higher temperature => more likely to sample low probability tokens)
        if temperature != 1.0:
            logits = logits / temperature
        # Top-p/top-k filtering
        logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        # Sample
        token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        return token
    def top_k_top_p_filtering(
        self,logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
    ):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(
                max(top_k, min_tokens_to_keep), logits.size(-1)
            )  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        return logits
    def inference(self,inputs):
        audio,output_text=inputs
        batchsize=len(audio)
        #print(audio[0].shape[0])
        audio_len=[audio[i].shape[0]//320-1 for i in range(batchsize)]
        audio_len=[(i+9)//10 for i in audio_len]
        # # print(audio.shape)
        # for i in range(batchsize):
        #     audio[i]=audio[i][:audio_len[i]*10]

        #print(audio_len)

        with torch.no_grad():
            input_values = self.processor(audio,  return_tensors="pt", sampling_rate=16000, padding=True).input_values.to(self.device)
            input_values=input_values.float()
            outputs=self.audio_model(input_values,output_hidden_states=True)['hidden_states']
            outputs=outputs[-1]
            time_len=outputs.shape[1]
            padded_time_len=(time_len+9)//10*10
            # print(padded_time_len)
            outputs=F.pad(outputs,(0,0,0,padded_time_len-time_len))
            # print(outputs.shape)
            outputs=outputs.view(batchsize,-1,10240)

        # audio_inputs=padded_audio_input_values.contiguous().view(batchsize,padded_time_len//10,-1)
        audio_inputs=outputs.contiguous()
        # print(audio_inputs.shape)
        torch.cuda.empty_cache()
        audio_inputs=self.audio_embedding_last_Linear(audio_inputs).half()
        #print(audio_len)
        audio_lengths = torch.tensor(audio_len)

        x=[]
        len_x1=[]
        for i in range(batchsize):

            audio_input=audio_inputs[i,:audio_lengths[i],:]
            #print(text_input_pre.shape,audio_input.shape,text_input_post.shape)
            bos_emb=self.bos_emb.unsqueeze(0).to(self.device).detach()
            # eos_emb=self.eos_emb.unsqueeze(0).to(self.device).detach()
            input_x=torch.cat((bos_emb,audio_input),dim=0)
            x.append(input_x)
            #print(text_lengths[i],audio_lengths[i])
            len_x1.append(input_x.shape[0])
        #x为一个list，保存的是每个batch的input，每一个元素代表一个text和一个audio拼接的input
        #len_x为一个list，保存的是每个batch的input的长度
        padded_x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
        x = torch.stack(list(padded_x), dim=0)
        x=x.to(torch.bfloat16)
        
        
        # print(self.llama.dtype)
        # print(x.dtype)
        with torch.no_grad():
            outputs = self.llama.generate(
                inputs_embeds=x,
                do_sample=False,  # Disable random sampling
                num_beams=5,  # Use beam search with 5 beams
                num_return_sequences=1,
                max_length=100,
                pad_token_id=self.text_tokenizer.pad_token_id,
                eos_token_id=self.text_tokenizer.eos_token_id,
            )
        outputs=self.text_tokenizer.batch_decode(outputs,skip_special_tokens=True)
        print(outputs)
        return outputs

