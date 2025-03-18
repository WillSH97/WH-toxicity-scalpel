'''
I'm only doing this for like being overly organised really.

Stealing from here: https://huggingface.co/datasets/JeanKaddour/minipile

sampling down to 10% to not blow up computer

'''

from datasets import load_dataset
import random
import numpy as np

dataset = load_dataset("JeanKaddour/minipile", split="train")

random.seed(42)

sample=random.sample(dataset["text"], k=int(0.03*float(len(dataset)))) #sample 30k rows

sample_minipile_text = " ".join(sample)