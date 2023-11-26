'''
using fine-tuned GPT2 model to auto-regressively generate text
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
    model_path : str = field(default='gpt2')
    prompt : str = field(default='The cat')
    max_length : int = field(default=50)
    num_return_sequences : int = field(default=1)
    repetition_penalty : float = field(default=1.0)
    do_sample : bool = field(default=True)
    top_k : int = field(default=50)
    top_p : float = field(default=0.95)
    temperature : float = field(default=1.0)
    

@torch.no_grad()
def generate(args):
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    prompt = args.prompt
    prompt = tokenizer.encode(prompt, return_tensors='pt')
    prompt = prompt.to('cuda' if torch.cuda.is_available() else 'cpu')

    generated = model.generate(prompt, max_length=args.max_length, num_return_sequences=args.num_return_sequences, repetition_penalty=args.repetition_penalty, do_sample=args.do_sample, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)
    generated = generated.tolist()
    for g in generated:
        text = tokenizer.decode(g, clean_up_tokenization_spaces=True)
        print(text)


if __name__ == '__main__':
    parser = HfArgumentParser((MyArguments,))

    args = parser.parse_args_into_dataclasses()[0]

    generate(args)
