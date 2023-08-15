import torch
import numpy as np

from npy_append_array import NpyAppendArray
from tqdm import tqdm
from transformers import MBartModel, MBart50Tokenizer
from transformers import M2M100Model, M2M100Tokenizer

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_tokenizer_model(model_name, sl):
   model_dict = {"mBART": {"file_path": "./models/mbart-large-50-many-to-many-mmt", "funcs": [MBart50Tokenizer, MBartModel]}, 
                 "m2m100": {"file_path": "./models/m2m100_418M", "funcs": [M2M100Tokenizer, M2M100Model]}}
   
   tokenizer = model_dict[model_name]["funcs"][0].from_pretrained(model_dict[model_name]["file_path"], src_lang=sl)
   model = model_dict[model_name]["funcs"][1].from_pretrained(model_dict[model_name]["file_path"]).to(device)

   if model_name == "m2m100":
    model = model.get_encoder()

   return tokenizer, model

def pool_embeddings(data, tokenized, pad_tok_id):
  if "attention_mask" in tokenized:
    attention_mask = tokenized["attention_mask"]
  else: # apparently ErnieM does NOT have attenion IDs in the tokenized output, so I am "computing" them myself - like in all other models, the model should not pay attention to [PAD] tokens, so they are ignored/not paid attention to
    token_ids = tokenized["input_ids"][0]
    padding_ids = len([tok for tok in token_ids if tok == pad_tok_id]) # count how many [PAD] tokens there are
    attention_mask = torch.ones((tokenized["input_ids"].shape)).to(device)
    if padding_ids > 0:
        attention_mask[:,-padding_ids:] = 0
    attention_mask = torch.tensor(attention_mask).to(device)
    
  attention_expanded = attention_mask.unsqueeze(-1).expand(data.size()).float()
  data_attention = data * attention_expanded
  return torch.sum(data_attention, 1) / torch.clamp(attention_expanded.sum(1), min=1e-9) # to not divide by 0

def get_embeddings(model_name, df, sl, extra):
    tokenizer, model = load_tokenizer_model(model_name, sl)
    extra = "" if len(extra) < 1 else "_" + extra

    embeddings = NpyAppendArray("data/" + f"emb_{model_name}_{sl}{extra}.npy", delete_if_exists=True)

    for claim in tqdm(df["claim"]):
      inputs = tokenizer(claim, return_tensors="pt")
      inputs = inputs.to(device)
      outputs = model(**inputs)

      pad_tok_id = tokenizer("<pad>")
      pad_tok_id = pad_tok_id["input_ids"][1]

      outputs = pool_embeddings(outputs[0], inputs, pad_tok_id)

      embeddings.append(np.array(outputs.cpu().detach().numpy()))