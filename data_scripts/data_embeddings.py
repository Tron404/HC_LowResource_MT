import pandas as pd
import os
import multiprocessing as mp

from emb import *

model_langs = {"mBART": ["ta_IN", "vi_VN", "xh_ZA"], "m2m100": ["ta", "vi", "xh"]}
data_path = "translation/complete/"

print(torch.cuda.is_available())

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    processes = []
    for file in os.listdir(data_path):
        for model, langs in model_langs.items():
            for lang in langs:
                sl = lang
                if model in file and lang.split('_')[0] in file.split("_"):
                    extra = ""
                    if "backtranslation" in file:
                        extra= f"backtranslation-{lang}" 
                        sl = "en_XX" if model == "mBART" else "en"
                    df = pd.read_csv(data_path + file, sep="|")
                    df["label"].to_csv("data/labels.csv", index=False)
                    # df = df.iloc[:20]
                    p = mp.Process(target=get_embeddings, args=(model, df, sl, extra))
                    print(f"Starting process for {model} | {sl} | {extra} | {df.shape}")
                    p.start()
                    processes.append(p)
        if len(processes) >= 6:
            print("Joining processes")
            for p in processes:
                p.join()

            processes = []
