import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, AdamW, get_scheduler
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from transformers import AdamW

from transformers import get_scheduler

from tqdm.auto import tqdm

model_name = 'microsoft/deberta-v3-base'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

raw_data = pd.read_csv('../datasets/biasly/biasly_prepared_df.csv')

len(raw_data)

#add data via pandas
tokenized_data = pd.DataFrame([tokenizer(x) for x in raw_data["datapoint"]])

total_data = pd.concat((raw_data, tokenized_data), axis=1)

total_data["labels"] = total_data["misogynistic_label"]


class BiaslyDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts = dataframe['datapoint'].tolist()
        self.labels = dataframe['misogynistic_label'].tolist()
        
        # Tokenize in the constructor
        self.encodings = tokenizer(self.texts, truncation=True, padding=True, max_length=512)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


whole_dataset = BiaslyDataset(raw_data, tokenizer)

# Split Dataset
generator = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(whole_dataset))
val_size = int(0.1 * len(whole_dataset))
test_size = len(whole_dataset) - train_size - val_size

train_set, val_set, test_set = torch.utils.data.random_split(
    whole_dataset, 
    [train_size, val_size, test_size], 
    generator=generator
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
batch_size = 4
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_epochs = 15


train_dataloader = DataLoader(
    train_set, shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    val_set, batch_size=batch_size, collate_fn=data_collator
)

# HYPERPARAMTETERS





num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

progress_bar = tqdm(range(num_training_steps))

train_batch_loss = []
val_batch_loss = []

train_epoch_loss = []
val_epoch_loss = []

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        train_batch_loss.append(float(loss))
        #eval per batch?
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        # batch eval loss ---? required?
        model.eval()
        temp_val_batch_loss=[]
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            temp_val_loss = outputs.loss
            temp_val_batch_loss.append(float(temp_val_loss))
        val_batch_loss = np.mean(temp_val_batch_loss)
        train_batch_loss.append(val_batch_loss)

        ### ADD PRINT STATEMENT HERE
        print(f"EPOCH: {epoch} --- train loss: {loss} --- val loss: {val_batch_loss}")
        
    # eval per epoch ???
    model.eval()
    temp_train_epoch_loss = []
    temp_val_epoch_loss = []
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        temp_train_batch_loss.append(float(loss))

    train_epoch_loss_value = np.mean(temp_train_epoch_loss)
    train_epoch_loss.append(train_epoch_loss_value)
        
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        temp_val_batch_loss.append(float(loss))

    val_epoch_loss_value = np.mean(temp_val_batch_loss)
    val_epoch_loss.append(val_epoch_loss_value)

    print("~~~~~~~~~OVERALL EPOCH LOSS~~~~~~~~~")
    print(f"EPOCH: {epoch} --- train loss: {train_epoch_loss_value} --- val loss: {val_epoch_loss_value}")


    
    