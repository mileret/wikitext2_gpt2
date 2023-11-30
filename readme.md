# Installation

Run `pip install -r requirement.txt` to prepare the environment.

To avoid unpredictable mistakes, you can also install the required packages by yourself. The main packages required are python=3.8, torch=2.1.0, transformers, datasets...

# Inference
Run following codes to inference with the given prompt.

```
cd codes
python inference.py --model_path path/to/your/checkpoint --prompt your/prompt
```

# Evaluation

Run the following codes to evaluate the checkpoints using Perplexity metric.

```
cd codes
python evaluation.py --ckpt filename/of/your/checkpoint
```


# Train

Run the following codes to finetune the pretrained GPT-2 model placed in `./pretrain`.

```
cd codes
python train.py
```

The training results will be placed under `./ckpts`.

# Reference

[code base 1](https://zhuanlan.zhihu.com/p/647862375?utm_id=0)

[code base 2](https://www.kaggle.com/code/changyeop/how-to-fine-tune-gpt-2-for-beginners)

[code base 3](https://github.com/huggingface/transformers/issues/9648)

[code base 4](https://github.com/mileret/nlpdl-hw3)
