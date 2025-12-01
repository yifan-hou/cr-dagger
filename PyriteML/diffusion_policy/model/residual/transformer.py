import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Callable, Any 
from torch.distributions import Normal

def convert_to_relative(time_embeds, reference_time=None):
    time_embeds_list = []
    for key in time_embeds:
        time_embeds_list.append(time_embeds[key] - reference_time)

    return torch.cat(time_embeds_list, dim=1)

class PositionalEncoding(nn.Module):
    """
    Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(self, embed_dim):
        """
        Standard sinusoidal positional encoding scheme in transformers.

        Positional encoding of the k'th position in the sequence is given by:
            p(k, 2i) = sin(k/n^(i/d))
            p(k, 2i+1) = sin(k/n^(i/d))

        n: set to 10K in original Transformer paper
        d: the embedding dimension
        i: positions along the projected embedding space (ranges from 0 to d/2)

        Args:
            embed_dim: The number of dimensions to project the timesteps into.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Input timestep of shape BxT
        """
        position = x

        # computing 1/n^(i/d) in log space and then exponentiating and fixing the shape
        div_term = (
            torch.exp(
                torch.arange(0, self.embed_dim, 2, device=x.device)
                * (-math.log(10000.0) / self.embed_dim)
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(x.shape[0], x.shape[1], 1)
        )
        pe = torch.zeros((x.shape[0], x.shape[1], self.embed_dim), device=x.device)
        pe[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
        return pe.detach()
    

class Transformer(nn.Module):
    '''
    Basic transformer implementation using torch.nn. Also added option for custom pos embeddings. 
    Made to be as basic as possible but also flexible to be put into ACT.

        d: hidden dimension
        h: number of heads
        d_ff: feed forward dimension
        num_layers: number of layers for encoder and decoder
        L: sequence length
        dropout: dropout rate
        src_vocab_size: size of source vocabulary
        tgt_vocab_size: size of target vocabulary
        pos_encoding_class : nn.Module class defining custom pos encoding

    '''
    def __init__(
        self,
        d : int,
        h : int,
        d_ff : int,
        num_layers : int,
        a: int,
        horizon: int,
        dropout : float = 0.1,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        pos_encoding_class: Optional[Callable[..., nn.Module]] = None,
        **pos_encoding_kwargs: Any  # Additional arguments for the custom encoding class
    ):
        super(Transformer, self).__init__()

        self.d = d
        self.h = h
        self.src_embed = nn.Embedding(src_vocab_size, d) if src_vocab_size else None
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d) if tgt_vocab_size else None
        self.a = a

        if pos_encoding_class is not None:
            self.src_pos_encoding = pos_encoding_class(**pos_encoding_kwargs)
            self.tgt_pos_encoding = pos_encoding_class(**pos_encoding_kwargs)
        else:
            self.src_pos_encoding = PositionalEncoding(d)
            self.tgt_pos_encoding = PositionalEncoding(d)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=h, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=h, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        
        # self.fc = nn.Linear(d, tgt_vocab_size) if tgt_vocab_size else None
        self.dropout = nn.Dropout(dropout)
        
        self.action_head = nn.Linear(d, self.a)
        
        self.time_embedding_layer = nn.Linear(1, d)
        

    def forward(self, src, time_embeds, target_shape, auto_masks=False):
        assert auto_masks == False, "Auto mask not supported"
        src_mask = tgt_mask = None
        
        tokens = []
        time_embeds_dict = dict()
        for key, attr in src.items():
            if "rgb" in key or 'time' in key:
                id = int(key.split("_")[-1])
            elif "policy" in key:
                id = int(key[12])
            else:
                id = int(key[5])  # robot0_xxxx

            if "rgb" in key:
                time_stamp_key = f"rgb_time_stamps_{id}"
            elif "wrench" in key:
                time_stamp_key = f"wrench_time_stamps_{id}"
            elif "gripper" in key:
                time_stamp_key = f"gripper_time_stamps_{id}"
            elif "policy" in key:
                time_stamp_key = f"policy_time_stamps_{id}"
            else:
                time_stamp_key = f"robot_time_stamps_{id}"
            if "wrench" in key:
                time_embeds_dict[key] = time_embeds[time_stamp_key][:, -1].unsqueeze(1)
            else:
                time_embeds_dict[key] = time_embeds[time_stamp_key]
            tokens.append(attr)
        src = torch.cat(tokens, dim=1)
        reference_time = time_embeds["policy_time_stamps_0"][:, 0].unsqueeze(1)
        time_embeds = convert_to_relative(time_embeds_dict, reference_time)
        src += self.time_embedding_layer(time_embeds.unsqueeze(-1))

        query_embed = nn.Embedding(target_shape[1], src.shape[2])
        query_embed = query_embed.weight.unsqueeze(0).repeat(target_shape[0], 1, 1)
        tgt = query_embed.to(src.device)

        if self.src_embed:
            src = self.src_embed(src)
        if self.tgt_embed:
            tgt = self.tgt_embed(tgt)


        src = src.transpose(0, 1)  # [sequence_length, batch_size, hidden_dim]
        src_position_indices = torch.arange(src.size(0), device=src.device).unsqueeze(1).expand(-1, src.size(1))  # [sequence_length, batch_size]
        src_pos = self.src_pos_encoding(src_position_indices)  # [sequence_length, batch_size, hidden_dim]
        src = src + src_pos 
        src = src.transpose(0, 1)

        tgt = tgt.transpose(0, 1)  # [T, B, hidden_dim]
        tgt_position_indices = torch.arange(tgt.size(0), device=tgt.device).unsqueeze(1).expand(-1, tgt.size(1))  # [sequence_length, batch_size]
        tgt_pos = self.tgt_pos_encoding(tgt_position_indices)
        tgt = tgt + tgt_pos 
        tgt = tgt.transpose(0, 1)  # [B, T, hidden_dim]

        src = self.dropout(src)
        tgt = self.dropout(tgt)

        enc = self.encoder(src, src_key_padding_mask=src_mask)
        dec = self.decoder(
            tgt, enc, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask
        )

        if self.action_head:
            dec = self.action_head(dec)

        return dec
    
    def compute_loss(self, src, time_embed, target):
        target_shape = target.shape
        pred = self.forward(src, time_embed, target_shape)
        loss = F.mse_loss(pred, target)
        return loss
    
    def predict_actions(self, src, time_embed, target_shape):
        # target_shape = target.shape
        pred = self.forward(src, time_embed, target_shape)
        return pred
