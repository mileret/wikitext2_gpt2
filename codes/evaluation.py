'''
using perplexity as the evaluation metric to evaluate the fine-tuned model
'''

import os
import torch
import pdb
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, HfArgumentParser
from transformers import DataCollatorForLanguageModeling
from dataclasses import dataclass, field

from dataset import get_dataset


@torch.no_grad()
def compute_metrics(eval_pred):
    '''
    compute perplexity
    '''
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]
    loss = torch.nn.functional.cross_entropy(predictions, labels)
    perplexity = torch.exp(loss)
    return {'perplexity': perplexity}


if __name__ == '__main__':
    pretrain_path = '../pretrain'
    from_local = True

    # model = GPT2LMHeadModel.from_pretrained('../ckpts/checkpoint-1000')
    model = GPT2LMHeadModel.from_pretrained('../ckpts/checkpoint-1000')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrain_path if from_local else 'gpt2')
    dataset = get_dataset(from_local, pretrain_path)['test']

    def tokenize(examples):
        tokenized = tokenizer(examples['text'], padding=False, truncation=True, max_length=512)
        return tokenized
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
    
    dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, data_collator=data_collator, eval_dataset=dataset, compute_metrics=compute_metrics)
    result = trainer.evaluate()
    print(result)

