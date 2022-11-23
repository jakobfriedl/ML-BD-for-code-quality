import requests
import pandas as pd
import numpy as np
import yaml
from yaml.loader import SafeLoader
import time

start = time.time()

base_url = "https://raw.githubusercontent.com/smola/language-dataset/master/data/"
yml_file = "../../../data/language_data/dataset.yml"
out_file = "../../../data/language_data/10_language_dataset_with_groundtruth.csv"

with open(yml_file) as file:
    yml = yaml.load(file, Loader=SafeLoader)

g_t = []
data = []
i = 0
for file in yml["files"]:
    print(f"{i}: getting {file}")
    lang = str(yml["files"][str(file)]["annotations"]["vote"])

    if lang not in g_t:
        g_t.append(lang)

    r = requests.get(base_url+file)
    data.append([g_t.index(lang), lang, r.text])
    i += 1

df = pd.DataFrame(data, columns=["Ground truth", "Language", "Code"])

df.to_csv(out_file, index=True)
print(time.time() - start)

