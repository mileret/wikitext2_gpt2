'''
train the pre-trained GPT2 model using huggingface's transformers and datasets and trainer
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


@dataclass
class MyArguments(TrainingArguments):
    # overwrite default values
    output_dir : str = field(default='../ckpts')
    num_train_epochs : int = field(default=30)
    per_device_train_batch_size : int = field(default=64)
    per_device_eval_batch_size : int = field(default=64)
    warmup_steps : int = field(default=0)
    weight_decay : int= field(default=0.1)
    logging_dir : str = field(default='./logs')
    logging_steps : int  = field(default=10)
    evaluation_strategy : str = field(default='steps')
    eval_steps : int = field(default=10)
    save_steps : int = field(default=10)
    save_total_limit : int = field(default=1)
    load_best_model_at_end : bool = field(default=True)
    metric_for_best_model : str = field(default='eval_loss')
    greater_is_better : bool = field(default=False)
    disable_tqdm : bool = field(default=True)
    report_to : str = field(default='wandb')
    run_name : str = field(default='test')
    seed : int = field(default=2022)
    learning_rate : float = field(default=1e-5)

    pretrain_path : str = field(default='../pretrain')
    from_local : bool = field(default=True)


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


def train(args):
    '''
    train the pre-trained GPT2 model using huggingface's transformers and datasets and trainer
    '''

    dataset = get_dataset(from_local=args.from_local, local_path=args.pretrain_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrain_path if args.from_local else 'gpt2')
    model = GPT2LMHeadModel.from_pretrained(args.pretrain_path if args.from_local else 'gpt2')

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def tokenize(examples):
        tokenized = tokenizer(examples['text'], padding=False)
        return tokenized
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
    
    dataset['train'] = dataset['train'].map(tokenize, batched=True, batch_size=len(dataset['train']))
    dataset['validation'] = dataset['validation'].map(tokenize, batched=True, batch_size=len(dataset['validation']))

    trainer = Trainer(model=model, 
                      args=args, 
                      data_collator=data_collator, 
                      tokenizer=tokenizer,
                      train_dataset=dataset['train'], 
                      eval_dataset=dataset['validation'],
                      compute_metrics=compute_metrics
                      )
    
    trainer.train()
    trainer.save_model(args.output_dir)



if __name__ == "__main__":

    args = MyArguments()
    train(args)
    