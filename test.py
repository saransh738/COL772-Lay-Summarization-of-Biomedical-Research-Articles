import pandas as pd
import numpy as np
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
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import (
    get_peft_model,
    LoraConfig,
    PeftType,
)
from tqdm import tqdm
import sys
import os


def load_json(path):
    with open(path,encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

class CustomTestDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.text = self.data.article

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([text], max_length= self.source_len,truncation = True, pad_to_max_length=True,return_tensors='pt')
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long)
        }
    
def write_predictions_to_file(predictions, file_path):
    with open(file_path, 'w') as file:
        for prediction in predictions:
            prediction_text = '\n'.join(prediction)  
            file.write(prediction_text + '\n')  


device = 'cuda' if cuda.is_available() else 'cpu'
torch.manual_seed(42) 
np.random.seed(42) 
torch.backends.cudnn.deterministic = True

path_to_data = sys.argv[2]
path_to_model  = sys.argv[3]
path_to_result = sys.argv[4]

set_seed(42)
#test_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
test_tokenizer =  BioGptTokenizer.from_pretrained("microsoft/biogpt")
test_tokenizer.padding_side = "left"
#model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")


PLOS_test_data = load_json(os.path.join(path_to_data,'PLOS_test.jsonl'))
eLife_test_data = load_json(os.path.join(path_to_data,'eLife_test.jsonl'))
#test_data = PLOS_test_data+eLife_test_data

for i in range(len(PLOS_test_data)):
    sections = PLOS_test_data[i]['article'].split('\n')
    abstract = sections[0]
    abstract = abstract + "\nExplanation: "
    PLOS_test_data[i]['article'] = abstract
    
for i in range(len(eLife_test_data)):
    sections = eLife_test_data[i]['article'].split('\n')
    abstract = sections[0]
    abstract = abstract + "\nExplanation: "
    eLife_test_data[i]['article'] = abstract
      
PLOS_test_df = pd.DataFrame(PLOS_test_data)
eLife_test_df = pd.DataFrame(eLife_test_data)
print(PLOS_test_df.head(1))

PLOS_test_set = CustomTestDataset(PLOS_test_df, test_tokenizer,1024)
eLife_test_set = CustomTestDataset(eLife_test_df, test_tokenizer,1024)
test_params = {'batch_size':1,'shuffle': False,'num_workers': 0}
PLOS_test_loader = DataLoader(PLOS_test_set, **test_params)
eLife_test_loader = DataLoader(eLife_test_set, **test_params)
# with open('/kaggle/input/pickled-2-epoch-model/saved_model_2.pkl', 'rb') as f:
#     model = pickle.load(f)
#model = BioGptForCausalLM.from_pretrained(os.path.join(path_to_model,'saved_model_3_1000'))
model = torch.load(os.path.join(path_to_model,'saved_model_final'),map_location = device)
model = model.to(device)
model.eval()
predictions_PLOS = []
predictions_eLife = []
for step,batch in enumerate(tqdm(PLOS_test_loader)):
    mask = batch['source_mask'].to(device,dtype = torch.long)
    ids = batch['source_ids'].to(device, dtype = torch.long)
    generated_ids = model.generate(input_ids = ids,attention_mask = mask, penalty_alpha=0.5, top_k=6,repetition_penalty=2.5, max_new_tokens=100)
    gen_text = []
    for g in generated_ids:
        gen_text.append(test_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    #print(gen_text)
    predictions_PLOS.append(gen_text)
    
for step,batch in enumerate(tqdm(eLife_test_loader)):
    mask = batch['source_mask'].to(device,dtype = torch.long)
    ids = batch['source_ids'].to(device, dtype = torch.long)
    #generated_ids = model.generate(input_ids = ids,attention_mask = mask, penalty_alpha=4.5,max_new_tokens=100,repetition_penalty=2.5)
    generated_ids = model.generate(input_ids = ids,attention_mask = mask, penalty_alpha=0.5, top_k=6,repetition_penalty=2.5, max_new_tokens=100)
    gen_text = []
    for g in generated_ids:
        gen_text.append(test_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    predictions_eLife.append(gen_text)

    
pred_PLOS_file = os.path.join(path_to_result,'PLOS_pred.txt')
pred_eLife_file = os.path.join(path_to_result,'eLife_pred.txt')
write_predictions_to_file(predictions_PLOS,pred_PLOS_file)
write_predictions_to_file(predictions_eLife,pred_eLife_file)