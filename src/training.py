"""
Training utilities and loss/collator functions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.tokenizer import get_tokenizer


def PretrainingCollator():
    def collate_fn(batch):
        tokens = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
        return {"input_ids": tokens}
    return collate_fn


def SFTCollator(model_name="answerdotai/ModernBERT-base"):
    tokenizer = get_tokenizer(model_name)
    eos_token = tokenizer.eos_token_id

    def _collate_fn(batch):
        inputs = [torch.tensor(b["input_ids"]) for b in batch]
        query_masks = [torch.tensor(b["query_mask"]) for b in batch]

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, padding_value=eos_token, batch_first=True)
        query_masks = torch.nn.utils.rnn.pad_sequence(query_masks, padding_value=1, batch_first=True)
        
        return {"input_ids": inputs, "query_mask": query_masks}
        
    return _collate_fn


def compute_mask_loss(logits, labels, t, batch_size, seq_len):
    loss_func = nn.CrossEntropyLoss(reduction="none")
    num_classes = logits.shape[-1]
    
    loss = loss_func(logits.reshape(batch_size*seq_len, num_classes), labels.flatten())
    loss = loss.reshape(batch_size, seq_len) / t
    loss = loss.mean()
    
    return loss


def compute_sft_loss(logits, labels, t, query_mask, batch_size, seq_len):
    loss_func = nn.CrossEntropyLoss(reduction="none")
    num_classes = logits.shape[-1]
    
    loss = loss_func(logits.reshape(batch_size*seq_len, num_classes), labels.flatten())
    loss = loss.reshape(batch_size, seq_len) / t
    
    answer_lengths = query_mask.sum(dim=1, keepdim=True)
    answer_lengths = answer_lengths.clamp_min(1)  
    loss = loss / answer_lengths
    loss = loss.sum(dim=1).mean()
    
    return loss
