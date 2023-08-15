import sys
import pandas as pd
import numpy as np

import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_path = "models/"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())

## mBART
# sl = "en_XX"
# tl_Ta = "ta_IN"
# tl_Xh = "xh_ZA"
# tl_Vi = "vi_VN"

##M2M100
sl = "en"
tl_Ta = "ta"
tl_Xh = "xh"
tl_Vi = "vi"

tls = [tl_Ta, tl_Xh, tl_Vi]

data_path = "data/data.csv"

def preprocess(text, tokenizer, tl, model):
  inputs = tokenizer(text, truncation=True, max_length=200, padding="max_length", return_tensors="pt")
  inputs = inputs.to(device)
  # generated_tok = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tl])
  generated_tok = model.generate(**inputs, forced_bos_token_id = tokenizer.get_lang_id(tl))

  return tokenizer.batch_decode(generated_tok, skip_special_tokens=True)[0]

def vec_preprocess(df_text, tokenizer, tl, model):
  print("Started processing using vectorised function")
  return np.vectorize(preprocess)(df_text, tokenizer, tl, model)

def translate(sl, tl, model, data_limit, id):
  print(f"{id} has entered")
  print(f"{sl} - {tl}")
  # tokenizer_SlTl = MBart50Tokenizer.from_pretrained(model_path + "mbart-large-50-many-to-many-mmt", src_lang=sl, tgt_lang=tl)
  tokenizer_SlTl = M2M100Tokenizer.from_pretrained(model_path + "m2m100_418M", src_lang=sl, tgt_lang=tl)

  df = pd.read_csv(data_path, sep="|")
  df = df.iloc[data_limit[0]:data_limit[1]]

  print(f"English --> {tl}")
  df["claim"] = vec_preprocess(df["claim"], tokenizer_SlTl, tl, model)

  df.to_csv(f"translation/data_translated_{tl}_{id}_m2m100.csv", encoding="utf-8", index=None, sep="|")

#SBATCH --array=1-7,9-11
def back_translate(sl, tl, model, data_limit, id):
  print(f"{id} has entered")
  # tokenizer_TlSl = MBart50Tokenizer.from_pretrained(model_path + "mbart-large-50-many-to-many-mmt", src_lang=tl, tgt_lang=sl)
  tokenizer_TlSl = M2M100Tokenizer.from_pretrained(model_path + "m2m100_418M", src_lang=tl, tgt_lang=sl)

  df = pd.read_csv(f"translation/data_translated_{tl}_{id}_m2m100.csv", sep="|")

  print(df)

  print(f"{tl} --> English")
  df["claim"] = vec_preprocess(df["claim"], tokenizer_TlSl, sl, model)

  df.to_csv(f"translation/data_translated_backtranslation_{tl}-{sl}_{id}_m2m100.csv", encoding="utf-8", index=None, sep="|")

