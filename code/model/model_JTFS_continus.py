import lightning.pytorch as pl
from typing import Tuple, Union
from transformers import AutoTokenizer, AutoModelForMaskedLM
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
from module.utils import make_pad_mask, Transpose
from transformers import WhisperProcessor, WhisperModel

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
        whisper_ckpt_path: str = None,
        text_bpe_ckpt_path: str = None,
        ):
        super(IS, self).__init__()
        self.processor=WhisperProcessor.from_pretrained(whisper_ckpt_path)
        self.model=WhisperModel.from_pretrained(whisper_ckpt_path)
        self.text_tokenizer=AutoTokenizer.from_pretrained(text_bpe_ckpt_path)
        self.new_token=["[/Begin]","[/End]"]
        self.text_tokenizer.add_tokens(self.new_token)
        for param in self.parameters():
            param.requires_grad = False
        self.text_embedding_model=nn.Embedding(len(self.text_tokenizer), 1024)
        self.norm_first=True
        self.audio_embedding_last_Linear=nn.Linear(1024*10,d_model)
        self.project_layer=nn.Linear(d_model,len(self.text_tokenizer))

        #【[GPT_Begin]8  [Text_Begin]text[Text_End] [Audio_Begin]Audio[Audio_End] [GPT_End]9】NULL
        #7个token，最后一个表示其他无关的tts表征
        self.num_heads=nhead

        self.decoder = decoder_cls(
                    decoder_layer_cls(
                        d_model,
                        nhead,
                        dim_feedforward=d_model * 4,
                        dropout=0.5,
                        batch_first=True,
                        norm_first=self.norm_first,
                    ),
                    num_layers=num_layers,
                    norm=LayerNorm(d_model) if self.norm_first else None,
                )
        
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

    def forward(self,inputs):
        audio,output_text=inputs
        batchsize=len(audio)
        #audio[i].shape[0]//320 的结果然后除以10的上界为len_audio
        #len_audio=[audio[i].shape[0]//320 for i in range(batchsize)]
        audio_len=[audio[i].shape[0]//320 for i in range(batchsize)]
        audio_len=[(i+9)//10 for i in audio_len]

        with torch.no_grad():
            input_values = self.processor(audio,  return_tensors="pt", sampling_rate=16000).input_features.to(self.device)
            outputs=self.model.get_encoder()(input_values)[-1]

        # audio_inputs=padded_audio_input_values.contiguous().view(batchsize,padded_time_len//10,-1)
        audio_inputs=outputs.contiguous().view(batchsize,150,-1)
        torch.cuda.empty_cache()
        audio_inputs=self.audio_embedding_last_Linear(audio_inputs)
        #print(audio_len)
        audio_lengths = torch.tensor(audio_len)

        x=[]
        len_x1=[]
        for i in range(batchsize):
            text_pre=self.new_token[0]
            text_post=self.new_token[1]
            text_pre=self.text_tokenizer(text_pre,return_tensors="pt")["input_ids"].to(self.device)
            text_post=self.text_tokenizer(text_post,return_tensors="pt")["input_ids"].to(self.device)
            text_pre=text_pre[:,:-1]
            text_post=text_post[:,:-1]
            text_input_pre=self.text_embedding_model(text_pre).squeeze(0)
            text_input_post=self.text_embedding_model(text_post).squeeze(0)
            audio_input=audio_inputs[i,:audio_lengths[i],:]
            #print(text_input_pre.shape,audio_input.shape,text_input_post.shape)
            input_x=torch.cat((text_input_pre,audio_input,text_input_post),dim=0)
            x.append(input_x)
            #print(text_lengths[i],audio_lengths[i])
            len_x1.append(input_x.shape[0])
        #x为一个list，保存的是每个batch的input，每一个元素代表一个text和一个audio拼接的input
        #len_x为一个list，保存的是每个batch的input的长度
        padded_x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
        x = torch.stack(list(padded_x), dim=0)
        #x的维度为【batchsize，len，d_model】
        len_y1=[]
        y=[]

        for i in range(batchsize):
            target_t=self.new_token[0]+output_text[i]+self.new_token[1]
            target_t=self.text_tokenizer(target_t,return_tensors="pt")["input_ids"].to(self.device)
            target_t=target_t[:,:-1].squeeze(0)
            
            len_y1.append(target_t.shape[0]-1)
            y.append(target_t)
        padded_y = rnn_utils.pad_sequence(y, batch_first=True, padding_value=0)
        y = torch.stack(list(padded_y), dim=0).to(x.device)


        len_x=torch.tensor(len_x1).clone().detach()
        len_y=torch.tensor(len_y1).clone().detach()

        x_len=len_x.max()
        y_len=len_y.max()
        #print(x_len,y_len)
        x_mask=make_pad_mask(len_x).to(x.device)
        y_mask=make_pad_mask(len_y).to(x.device)
        y_mask_int = y_mask.type(torch.int64)

        xy_padding_mask = torch.cat([x_mask, y_mask], dim=1)
        #xy_padding_mask的维度为【batchsize，len】
        

        x_attn_mask=F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.cat([x_attn_mask, y_attn_mask], dim=0)

        x_attn_mask=None
        y_attn_mask=None
        torch.cuda.empty_cache()

        # merge key padding and attention masks
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(bsz * self.num_heads, 1, src_len)
        )

        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        #print(xy_attn_mask.shape)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))

        _xy_padding_mask=None
        new_attn_mask=None
        torch.cuda.empty_cache()

        y_in, targets = y[:, :-1], y[:, 1:]
        y_in=y_in.to(x.device)
        targets=targets.to(x.device)
        y_in=self.text_embedding_model(y_in)
        
        tgt_mask = torch.triu(
            torch.ones(y_len, y_len, device=y.device, dtype=torch.bool),
            diagonal=1,
        )
        # print("y_in:",y_in.shape)
        # print("x:",x.shape)
        # print("tgt_mask:",tgt_mask.shape)
        # print("y_mask:",y_mask.shape)
        # print("x_mask:",x_mask.shape)
        y_dec, _ = self.decoder(
            (y_in, None),
            x,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=y_mask,
            memory_mask=None,
            memory_key_padding_mask=x_mask,
        )
        #y_dec和y_in的mseloss
        mask_y=make_pad_mask(len_y-1).to(y.device)
        mask_y=mask_y.unsqueeze(1).expand(-1,y_dec.shape[2],-1).permute(0,2,1)
        # print(y_dec.shape,y_in.shape)
        # print(mask_y.shape)
        y_dec_mse=y_dec[:,:-1,:]*~mask_y
        y_in_mse=y_in[:,1:,:]*~mask_y
        loss_ar=nn.MSELoss()(y_dec_mse,y_in_mse)
        # loss_ar=F.mse_loss(y_dec
        logits=self.project_layer(y_dec).permute(0, 2, 1)
        # print(logits.shape)
        #[4,11035,1122]
        #print(logits)
        #print(targets.shape)
        #print(targets[0][:100])

        top1acc=self.accuracy1(
            logits.detach(), targets
        ).item() * len_y.sum().type(torch.float32)
        top1acc=top1acc/len_y.sum()

        torch.cuda.empty_cache()

        top5acc=self.accuracy5(
            logits.detach(), targets
        ).item() * len_y.sum().type(torch.float32)
        top5acc=top5acc/len_y.sum()
        #print("top5acc",top5acc)

        torch.cuda.empty_cache()
        
        top10acc=self.accuracy10(
            logits.detach(), targets
        ).item() * len_y.sum().type(torch.float32)
        top10acc=top10acc/len_y.sum()

        torch.cuda.empty_cache()
        
        total_loss = F.cross_entropy(logits, targets, reduction="mean")
        total_loss+=loss_ar*100

           
        return (total_loss,top1acc)
    def training_step(self,batch,batch_idx):
        loss,top1acc=self.forward(batch)
        if torch.isnan(loss):
            print("loss is nan")
            gradients = [p.grad.detach().cpu().numpy() for p in self.parameters() if p.grad is not None]
            print("Gradients:", gradients)
            with open("wrong.txt","w") as f:
                f.write("loss is nan")
                f.write(str(gradients))
            exit()
        self.log("asr_loss", loss, on_step=True, on_epoch=True, prog_bar=True,batch_size=len(batch[0]),sync_dist=True)
        self.log("asr_top1acc", top1acc, on_step=True, on_epoch=True, prog_bar=True,batch_size=len(batch[0]),sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,batch_size=len(batch[0]),sync_dist=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
        return optimizer
    def validation_step(self,batch,batch_idx):
        loss,top1acc=self.forward(batch)
        self.log("asr_val_loss", loss, on_step=True, on_epoch=True, prog_bar=True,batch_size=len(batch[0]),sync_dist=True)
        self.log("asr_val_top1acc", top1acc, on_step=True, on_epoch=True, prog_bar=True,batch_size=len(batch[0]),sync_dist=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True,batch_size=len(batch[0]),sync_dist=True)
        return loss
    

    def test_asr(self,inputs,filedir):
        y=self.inference(inputs)
        y_pre=y[0]
        # print("y_pre:")
        # print(y_pre)
        try:
            result = y_pre[1:]
            output=self.text_tokenizer.decode(result)
            # print("output:")
            # print(output)
            # print("target:")
            # print(inputs[1])
            # print("\n")
            target_dir=os.path.join(filedir,"target.txt")
            output_dir=os.path.join(filedir,"output.txt")
            if not os.path.exists(target_dir):
                with open(target_dir,"w") as f:
                    pass
            if not os.path.exists(output_dir):
                with open(output_dir,"w") as f:
                    pass
            with open(target_dir,"a") as f:
                # print(inputs[1][0])
                # print(1)
                f.write(inputs[1][0]+"\n")
            with open(output_dir,"a") as f1:
                # print(2)
                f1.write(output+"\n")
            
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
        #audio[i].shape[0]//320 的结果然后除以10的上界为len_audio
        #len_audio=[audio[i].shape[0]//320 for i in range(batchsize)]
        audio_len=[audio[i].shape[0]//320 for i in range(batchsize)]
        audio_len=[(i+9)//10 for i in audio_len]

        with torch.no_grad():
            input_values = self.processor(audio,  return_tensors="pt", sampling_rate=16000).input_features.to(self.device)
            outputs=self.model.get_encoder()(input_values)[-1]

        # audio_inputs=padded_audio_input_values.contiguous().view(batchsize,padded_time_len//10,-1)
        audio_inputs=outputs.contiguous().view(batchsize,150,-1)
        torch.cuda.empty_cache()
        audio_inputs=self.audio_embedding_last_Linear(audio_inputs)
        #print(audio_len)
        audio_lengths = torch.tensor(audio_len)

        x=[]
        len_x1=[]
        for i in range(batchsize):
            text_pre=self.new_token[0]
            text_post=self.new_token[1]
            text_pre=self.text_tokenizer(text_pre,return_tensors="pt")["input_ids"].to(self.device)
            text_post=self.text_tokenizer(text_post,return_tensors="pt")["input_ids"].to(self.device)
            text_pre=text_pre[:,:-1]
            text_post=text_post[:,:-1]
            text_input_pre=self.text_embedding_model(text_pre).squeeze(0)
            text_input_post=self.text_embedding_model(text_post).squeeze(0)
            audio_input=audio_inputs[i,:audio_lengths[i],:]
            #print(text_input_pre.shape,audio_input.shape,text_input_post.shape)
            input_x=torch.cat((text_input_pre,audio_input,text_input_post),dim=0)
            x.append(input_x)
            #print(text_lengths[i],audio_lengths[i])
            len_x1.append(input_x.shape[0])
        #x为一个list，保存的是每个batch的input，每一个元素代表一个text和一个audio拼接的input
        #len_x为一个list，保存的是每个batch的input的长度
        padded_x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
        x = torch.stack(list(padded_x), dim=0)
        
        len_x=torch.tensor(len_x1).clone().detach()

        x_mask = make_pad_mask(len_x).to(x.device)

        begin_token=self.new_token[0]
        end_token=self.new_token[1]
        begin_token=self.text_tokenizer(begin_token,return_tensors="pt")["input_ids"][0][0].to(self.device)
        end_token=self.text_tokenizer(end_token,return_tensors="pt")["input_ids"][0][0].to(self.device)

        y=begin_token.unsqueeze(0).unsqueeze(0)
        y_pos=self.text_embedding_model(y)
        while True:
            tgt_mask = torch.triu(
                    torch.ones(
                        y_pos.shape[1], y_pos.shape[1], device=y_pos.device, dtype=torch.bool
                    ),
                    diagonal=1,
            )
            with torch.no_grad():
                y_dec, _ = self.decoder(
                    (y_pos, None),
                    x,
                    tgt_mask=tgt_mask,
                    memory_mask=None,
                    memory_key_padding_mask=x_mask,
                )
            logits=self.project_layer(y_dec[:,-1])
            samples=self.topk_sampling(
                logits, top_k=1, top_p=1.0, temperature=1.0
            )
            if(samples[0,0]==end_token or (y.shape[1]) >len_x.max() * 64):
                break
            y = torch.concat([y, samples], dim=1)
            y_pos=torch.cat([y_pos,y_dec[:,-1].unsqueeze(1)],dim=1)

        return y

