import requests
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import time
from preprocessing import process_stack_trace_column

start = time.time()

base_url = "https://raw.githubusercontent.com/smola/language-dataset/master/data/"
yml_file = "../../../data/language_data/dataset.yml"
out_file = "../../../data/language_data/language_dataset.csv"

with open(yml_file) as file:
    yml = yaml.load(file, Loader=SafeLoader)

data = []
for file in yml["files"]:
    print("getting {}".format(file))
    lang = str(yml["files"][str(file)]["annotations"]["vote"])
    r = requests.get(base_url+file)
    data.append([lang, r.text])

df = pd.DataFrame(data, columns=["Language", "Code"])
df.to_csv(out_file, index=False)
print(time.time() - start)

