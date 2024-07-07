import pandas
import json
import torch
from torch import cuda
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed
from transformers import Adafactor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup
import pickle
from peft import (
    get_peft_model,
    LoraConfig,
    PeftType,
)
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd

def load_json(path):
    with open(path,encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.article
        self.target_text = self.data.lay_summary

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.target_text[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([text], max_length= self.source_len,truncation = True, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([ctext], max_length= self.summ_len,truncation = True, pad_to_max_length=True,return_tensors='pt')
        target["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in target["input_ids"]]
        target["input_ids"] = torch.tensor(target["input_ids"])
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_ids.to(dtype=torch.long)
        }

path_to_data = sys.argv[2]   
config = config = AutoConfig.from_pretrained("microsoft/biogpt", cache_dir=None)
config.TRAIN_BATCH_SIZE = 2   
config.VALID_BATCH_SIZE = 1   
config.TRAIN_EPOCHS = 4        
config.VAL_EPOCHS =  1
config.LEARNING_RATE = 2e-4    
config.SEED = 42               
config.MAX_LEN = 1024
config.SUMMARY_LEN = 1024

device = 'cuda' if cuda.is_available() else 'cpu'
torch.manual_seed(config.SEED) 
np.random.seed(config.SEED) 
torch.backends.cudnn.deterministic = True

set_seed(42)
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
tokenizer.padding_side = "right"
val_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
val_tokenizer.padding_side = "left"
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
#with open("/home/maths/dual/mt6190837/mt6190837_a3/output/saved_model_2.pkl", 'rb') as f:
#model = torch.load("/home/maths/dual/mt6190837/mt6190837_a3/output/saved_model_3_10000",map_location = torch.device('cuda'))
#model.to(device)
PLOS_train_data = load_json(os.path.join(path_to_data,'PLOS_train.jsonl'))
eLife_train_data = load_json(os.path.join(path_to_data,'eLife_train.jsonl'))
PLOS_val_data = load_json(os.path.join(path_to_data,'PLOS_val.jsonl'))
eLife_val_data = load_json(os.path.join(path_to_data,'eLife_val.jsonl'))
train_data = PLOS_train_data+eLife_train_data
val_data = PLOS_val_data+eLife_val_data

for i in range(len(train_data)):
    sections = train_data[i]['article'].split('\n')
    abstract = sections[0]
    abstract = abstract + "\nExplanation: "
    train_data[i]['article'] = abstract
    
    
    
for i in range(len(val_data)):
    sections = val_data[i]['article'].split('\n')
    abstract = sections[0]
    abstract = abstract + "\nExplanation: "
    val_data[i]['article'] = abstract
    
train_df = pd.DataFrame(train_data)  
val_df = pd.DataFrame(val_data)
print(train_df.head(1))


training_set = CustomDataset(train_df, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
val_set = CustomDataset(val_df, val_tokenizer, config.MAX_LEN, config.SUMMARY_LEN)


train_params = {'batch_size': config.TRAIN_BATCH_SIZE,'shuffle': True,'num_workers': 0}

val_params = {'batch_size': config.VALID_BATCH_SIZE,'shuffle': False,'num_workers': 0}


training_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(val_set, **val_params)


peft_type = PeftType.LORA
# Define LoRA Config
peft_config  = LoraConfig(
 r=8,
 lora_alpha=64,
 inference_mode=False,
 lora_dropout=0.1,
 target_modules= ['k_proj', 'v_proj', 'q_proj','fc1','fc2']
 #target_modules = [find_all_linear_names(model)]
 )
model = get_peft_model(model, peft_config)
model.to(device)
model.print_trainable_parameters()
#model
model = model.to(device)
optimizer = Adafactor(model.parameters(),weight_decay=0.01,lr = 2e-4,relative_step=False)

# for param_group in optimizer.param_groups:
#     param_group['lr'] = config.LEARNING_RATE

NUM_EPOCHS = 4
total_training_steps = (len(train_df)/config.TRAIN_BATCH_SIZE)*config.TRAIN_EPOCHS
if len(train_df)%config.TRAIN_BATCH_SIZE:
    total_training_steps+= config.TRAIN_EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.10 * (len(training_loader) * NUM_EPOCHS ),num_training_steps=(len(training_loader) * NUM_EPOCHS ))

print('Initiating Fine-Tuning for the model on dataset')

path_to_save = sys.argv[3]
file_val = open(os.path.join(path_to_save,"logs.txt"), "w")
#NUM_EPOCHS = 4
for epoch in tqdm(range(NUM_EPOCHS)):
    model.train()
    total_loss = 0
    val_loss = 0
    for step,batch in enumerate(tqdm(training_loader)):
        model.zero_grad()
        y = batch['target_ids'].to(device, dtype = torch.long)
        #input_mask = batch['source_mask'].to()
        ids = batch['source_ids'].to(device, dtype = torch.long)
        outputs = model(input_ids = ids,labels= y)
        loss = outputs[0]
        total_loss += loss.item()
        if step%10 == 0:
            print(f'Training Loss: {total_loss/(step+1)}')

        if step%500==0:
            print(f'Epoch: {epoch}, Loss:  {total_loss/(step+1)}')
            
        if step%10000 == 0:
            torch.save(model,os.path.join(path_to_save,"saved_model_final"))
                                            
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)  
        optimizer.step()  
        scheduler.step()
        if step%10000 == 0:
            model.eval()
            predictions = []
            actuals = []
            val_loss = 0
            for step_val,batch_val in enumerate(tqdm(val_loader)):
                y = batch_val['target_ids'].to(device, dtype = torch.long)
                mask = batch_val['source_mask'].to(device,dtype = torch.long)
                ids = batch_val['source_ids'].to(device, dtype = torch.long)
                with torch.no_grad():
                    outputs = model(input_ids = ids,labels= y)
                val_loss += outputs[0].item()
                #generated_ids = model.generate(input_ids = ids,attention_mask = mask, penalty_alpha=0.5, top_k=6, max_new_tokens=768)
                #predictions.append(tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids)
                #actuals.append(tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y)

            print(f'Completed Epoch:{epoch} val_loss:{val_loss}')
            #score_dict = evaluate(predictions,val_data)
            #f.write("Epoch : {} Val Loss: {:.4} BERTScore: {}  LENS: {} AlignScore {}  \n \n".format(epoch, val_loss,score_dict['BERTScore'],score_dict['LENS'],score_dict['AlignScore']))
            file_val.write("Epoch:{} Step:{} Val Loss: {:.4} \n \n".format(epoch,step,val_loss))
    #print(evaluate(predictions,val_data))
torch.save(model,os.path.join(path_to_save,"saved_model_final"))
file_val.close()
