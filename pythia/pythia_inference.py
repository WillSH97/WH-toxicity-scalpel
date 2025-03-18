from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch

def load_model(MODEL_DIR, TOKENIZER_DIR):

# MODEL_DIR = '../../misc-NOTGIT/pythiatest/toxicity-scalpel-supercompute/pythia-70m-deduped-finetuned-target-0.4' #obviously change this

    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_DIR,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, tokenizer, device


def pythia_generate(model, tokenizer, device, user_input, temperature=0.1, max_length=128):
    
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    tokens = model.generate(**inputs, do_sample=True,
        temperature=temperature,
        max_length=max_length,)
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return(output)
