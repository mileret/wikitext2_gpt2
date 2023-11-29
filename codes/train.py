'''
train the pre-trained GPT2 model using huggingface's transformers and datasets and trainer
'''

import os
import torch
import pdb

from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import GPT2Tokenizer,GPT2Model,AutoModel

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from dataclasses import dataclass, field

from dataset import get_dataset


def train():
    '''
    train the pre-trained GPT2 model using huggingface's transformers and datasets and trainer
    '''

    # load model and tokenizer
    pretrain_path = '../pretrain'
    from_local = True

    tokenizer = AutoTokenizer.from_pretrained(pretrain_path if from_local else 'gpt2')
    model = GPT2LMHeadModel.from_pretrained(pretrain_path if from_local else 'gpt2')

    max_seq_length = 512
    out_model_path = "../ckpts"
    train_epoches = 2
    batch_size = 16

    tokenizer.pad_token = tokenizer.eos_token

    # load and preprocess dataset
    dataset = get_dataset(from_local=True, local_path='../pretrain')
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    def tokenize(example):
        return tokenizer(example['text'], padding=False, truncation=True, max_length=max_seq_length)

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns='text')
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns='text')
    

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
            output_dir=out_model_path,
            overwrite_output_dir=True,
            num_train_epochs=train_epoches,
            per_device_train_batch_size=batch_size,
            save_steps=100,
            save_total_limit=1,
            prediction_loss_only=True,
            report_to=['tensorboard'],
            logging_steps=100,
        )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("--------------------------------------------------Begin Training--------------------------------------------------")

    trainer.train()
    trainer.save_model(out_model_path)

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


if __name__ == "__main__":

    train()
    