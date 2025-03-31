from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch

def load_model(MODEL_DIR, TOKENIZER_DIR):

# MODEL_DIR = '../../misc-NOTGIT/pythiatest/toxicity-scalpel-supercompute/pythia-70m-deduped-finetuned-target-0.4' #obviously change this

    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_DIR,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR, 
        padding_side='left',
    )
    
    device = 'cpu' # 'cuda' if torch.cuda.is_available() else <---- deleting this because I need to send this model to a bunch of devices anyway
    model.to(device)
    return model, tokenizer, device


def pythia_generate(model, tokenizer, device, user_input, temperature=0.1, max_new_tokens=128):
    
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    tokens = model.generate(**inputs, do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,)
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return(output)

### CLAUDE-GENERATED - looks fine and I'm lazy

def pythia_generate_batched(model, tokenizer, device, user_inputs, batch_size=32, temperature=0.1, max_new_tokens=128):
    """
    Generate text completions for multiple inputs in batches.
    
    Args:
        model: The GPTNeoXForCausalLM model
        tokenizer: The tokenizer for the model
        device: The device to run the model on ('cuda' or 'cpu')
        user_inputs: List of strings to generate completions for
        batch_size: Number of inputs to process at once
        temperature: Sampling temperature for generation
        max_length: Maximum length of generated sequences
        
    Returns:
        List of generated outputs corresponding to each input
    """
    outputs = []
    
    # Process in batches
    for i in range(0, len(user_inputs), batch_size):
        batch = user_inputs[i:i+batch_size]
        
        # Tokenize the batch
        batch_tokens = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        
        # Generate completions
        generated_tokens = model.generate(
            input_ids=batch_tokens.input_ids,
            attention_mask=batch_tokens.attention_mask,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        
        # Decode the generated tokens
        batch_outputs = [
            tokenizer.decode(tokens, skip_special_tokens=True) 
            for tokens in generated_tokens
        ]
        
        outputs.extend(batch_outputs)
    
    return outputs