import pandas as pd
from collections import defaultdict
import os
import sys
import pickle

from comet import download_model, load_from_checkpoint

model_qe_path = download_model("Unbabel/wmt20-comet-qe-da")
if not 'model_qe' in globals():
  model_qe = load_from_checkpoint(model_qe_path)

path = "translation/complete/"

df_orig = pd.read_csv("data/data.csv", sep="|")
original = [df_orig["claim"].to_numpy().tolist()]

df_dic_trans = defaultdict()
for file in os.listdir(path):
    if "backtranslation" not in file:
        df = pd.read_csv(path + file, sep="|")
        lang, model  = file.split("_")[2:4]
        model = model.split(".")[0]

        df_dic_trans[lang + "_" + model] = df["claim"].to_numpy().tolist()

from collections import defaultdict

df_orig = pd.read_csv("data/data.csv", sep="|")
original = df_orig["claim"].to_numpy().tolist()

model_df = defaultdict(list)

for key, val in df_dic_trans.items():
    for s, o in zip(val, original):
        model_df[key].append({"src": o, "mt": s})

aux = list(model_df.items())

id_input = int(sys.argv[1])

# model_output = model_qe.predict(aux[id_input][1], batch_size=24, gpus=1)

print(model_df.keys())

# print("-------------------------")
# print(model_output)
# print("-------------------------")