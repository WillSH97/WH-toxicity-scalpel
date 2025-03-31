'''
basing the dev of this off of: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-guard-2/

stealing this code: https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_llama_guard_model(device):
    model_id = "meta-llama/Meta-Llama-Guard-2-8B"
    # device = "cpu"
    dtype = torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 torch_dtype=dtype, 
                                                 load_in_4bit=False, #only need this for itty bitty devices like mine
                                                 device_map=device)
    return model, tokenizer, device

def moderate(input_str: str, model, tokenizer, device):
    chat = [{"role": "user", "content": input_str},]
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)