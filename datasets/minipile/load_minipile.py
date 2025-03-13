'''
I'm only doing this for like being overly organised really.

Stealing from here: https://huggingface.co/datasets/JeanKaddour/minipile

'''

from datasets import load_dataset

dataset = load_dataset("JeanKaddour/minipile", split="train")

minipile_text = dataset["text"]