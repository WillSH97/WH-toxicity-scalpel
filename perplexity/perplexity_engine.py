'''
intention here is to measure the perplexity of a model based on prob outputs compared to known human-generated text.

n.b. we're using some "ground truth" text dataset, and using a model and its measured prob dist outputs for each conditional chunk to calculate perplexity. We're not using a model like GPT-2 or MLMPPL like NLP people would.
(bit new for me but the logic tracks)

stealing a lot of code from here: https://huggingface.co/docs/transformers/en/perplexity
'''
from transformers import GPTNeoXForCausalLM, AutoTokenizer

import torch
from tqdm import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available else 'cpu'

def ppl(model, tokenizer, candidate_str: str):
    
    encodings = tokenizer(candidate_str, return_tensors="pt")
    
    max_length = model.config.max_position_embeddings #max context length
    stride = int(np.round(model.config.max_position_embeddings/2)) #half max context length
    seq_len = encodings.input_ids.size(1)
    
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
    
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
    
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
    
        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens
    
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return ppl

def ppl_from_dir(model_dir: str, candidate_str: str, quantize = False):
    model = GPTNeoXForCausalLM.from_pretrained(
    model_dir, ## NOTE: use whatever model path here
    device_map = device,
    load_in_4bit = quantize #for itty bitty machines like mine
)
    tokenizer = AutoTokenizer.from_pretrained(
    model_dir, ## NOTE: use whatever model path here
)
    ppl = ppl(model, tokenizer, candidate_str)
    return ppl