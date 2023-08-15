import sys
from translate import *

import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # model = MBartForConditionalGeneration.from_pretrained(model_path + "mbart-large-50-many-to-many-mmt").to(device)
    model = M2M100ForConditionalGeneration.from_pretrained(model_path + "m2m100_418M").to(device)

    data_limit = {0: [0, 4000], 1: [4000, 8000], 2: [8000, 12000], 3: [12000, 16000], 4: [16000, 20000], 5: [20000, 24000], 6: [24000, 28000], 7: [28000, 32000], 8: [32000, 36000], 9: [36000, 40000], 10: [40000, 44000], 11: [44000, 48000]}
    data_id = int(sys.argv[1])

    print(f"Starting job with id {data_id} and limits {data_limit[data_id]} - M2M")

    print("Beginning backtranslation process")
    process = []
    for idx, tl in enumerate(tls):
        p = mp.Process(target=back_translate, args=(sl, tl, model, data_limit[data_id], data_id))
        print(f"Starting process {idx+1}")
        p.start()
        process.append(p)

    for p in process:
        p.join()