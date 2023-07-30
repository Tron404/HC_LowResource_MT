import sys
import pandas as pd

import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

model_path = "models/"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())

idx = 0
max_len = -1

sl = "en_XX"
tl_Ta = "ta_IN"
tl_Xh = "xh_ZA"
tl_Vi = "vi_VN"

tls = [tl_Ta, tl_Xh, tl_Vi]
# tls = [tl_Vi]

data_path = "data/data_sampled.csv"

model = MBartForConditionalGeneration.from_pretrained(model_path + "mbart-large-50-many-to-many-mmt").to(device)

def preprocess(text, tokenizer, tl):
  inputs = tokenizer(text, truncation=True, max_length=200, padding="max_length", return_tensors="pt")
  inputs = inputs.to(device)
  generated_tok = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tl])

  global idx
  idx += 1

  if idx % 1000 == 0:
    print(f"{idx}/{max_len}")

  return tokenizer.batch_decode(generated_tok, skip_special_tokens=True)[0]

def translate(sl, tl):
  print(f"{sl} - {tl}")
  tokenizer_SlTl = MBart50Tokenizer.from_pretrained(model_path + "mbart-large-50-many-to-many-mmt", src_lang=sl, tgt_lang=tl)
  tokenizer_TlSl = MBart50Tokenizer.from_pretrained(model_path + "mbart-large-50-many-to-many-mmt", src_lang=tl, tgt_lang=sl)

  df = pd.read_csv(data_path, sep="|")
  print(df)

  global max_len
  max_len = len(df)

  global idx
  idx = 0
  df["claim"] = df["claim"].apply(preprocess, args=(tokenizer_SlTl, tl))
  df.to_csv(f"data_translated_{tl}.csv", encoding="utf-8", index=None)

  idx = 0
  df["claim"] = df["claim"].apply(preprocess, args=(tokenizer_TlSl, sl))
  df.to_csv(f"data_translated_backtranslation_{tl}-{sl}.csv", encoding="utf-8", index=None)
