import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import pandas as pd
import numpy as np

from torch.nn import Softmax

def load_deberta_finetune_model(dir_string: str):
    model_name = 'microsoft/deberta-v3-base'
    
    MODEL_WEIGHT_PATH=dir_string #replace with finetune model weight path
    
    #MODEL LOAD SCRIPTS
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, weights_only=True))
    
    model.to(device)
    
    model.eval()
    return model, tokenizer, device


m=Softmax()

class BiaslyInfDataset(Dataset):
    def __init__(self, str_list, tokenizer):
        self.texts = str_list
        
        # Tokenize in the constructor
        self.encodings = tokenizer(self.texts, truncation=True, padding=True, max_length=512)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


#FUNCS
def deberta_classify(model, tokenizer, device, input_str_list: str, batch_size: int = 32):
    '''
    simple func to return results

    inputs:
    - input_str_list (list(string)): the string to classify
    - batch_size (int): batch size for inference

    outputs 
    - results_list (list(int(bool))): 0 or 1, 1 being misogynistic classification, 0 being non-misogynistic.
    
    '''
    inputs_dataset = BiaslyInfDataset(input_str_list, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(inputs_dataset, 
                            batch_size=batch_size, 
                            collate_fn=data_collator)
    logit_results = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logit_results.extend(outputs.logits.tolist())

    softmax = m(torch.tensor(logit_results))
    #probs for ps class
    probs = np.array(softmax)[:,1].reshape(-1,1)
    
    #threshold at 0.75 for pos class
    results_list = [1 if x[0]>=0.75 else 0 for x in probs.tolist()]

    return results_list
