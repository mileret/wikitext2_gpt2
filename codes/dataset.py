'''
prepare wikitext-2 dataset from huggingface
'''
import os
import torch
import pdb
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset, load_from_disk


def get_dataset(from_local : bool, local_path : str):
    '''
    return wikitext-2 dataset from huggingface or local path
    '''
    
    if from_local:
        try:
            dataset = load_from_disk(local_path)
        except:
            raise ValueError(f'local_path {local_path} is not valid')
    else:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        
    
    # remove all the '' in the dataset
    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].filter(lambda x: len(x['text']) > 0)

    return dataset
    

# debug
if __name__ == "__main__":
    dataset = load_dataset('./wikitext.py', 'wikitext-2-raw-v1')
    pdb.set_trace()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


