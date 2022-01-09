from preprocessing import process_stack_trace_column
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df_comp = pd.read_csv('../../data/monkey_data_stack_trace_entry.csv')
df_monkey = pd.read_csv('../../data/monkey_data_stack_trace.csv')
df_github = pd.read_csv('../../data/github_issues_stack_trace.csv')


def add_values_in_dict(sample_dict, key, list_of_values):
    if key not in sample_dict:
        sample_dict[key] = list()
    sample_dict[key].append(list_of_values)
    return sample_dict


def spacy_word_2_vec(dataframe):
    print("""                                    
▀████▀     █     ▀███▀      ▀████▀   ▀███▀
  ▀██     ▄██     ▄█          ▀██     ▄█  
   ██▄   ▄███▄   ▄█   ███▀██▄  ██▄   ▄█   
    ██▄  █▀ ██▄  █▀  ███   ██   ██▄  █▀   
    ▀██ █▀  ▀██ █▀       ▄▄██   ▀██ █▀    
     ▄██▄    ▄██▄     ▄▄█▀       ▄██▄     
      ██      ██     ████████     ██      
    """)
    nlp = spacy.load('en_core_web_md')
    docs = dataframe[:].apply(lambda x: nlp(x))
    pdv = []
    for index, value in docs.iteritems():
        pdv.append(value.vector)
    return pdv


# Preprocessing
print("""
 ___              ___              
(  _ \           (  _ \            
| |_) )_ __   __ | |_) )_ __   _   
|  __/(  __)/ __ \  __/(  __)/ _ \ 
| |   | |  (  ___/ |   | |  ( (_) )
(_)   (_)   \____)_)   (_)   \___/ 
""")
data = process_stack_trace_column(df_monkey, 'l')
y = data.iloc[:].values

# Word Embedding
print("""
┏┓┏┓┏┓━━━━━━━━━┏┓┏━━━┓━━━━┏┓━━
┃┃┃┃┃┃━━━━━━━━━┃┃┃┏━━┛━━━━┃┃━━
┃┃┃┃┃┃┏━━┓┏━┓┏━┛┃┃┗━━┓┏┓┏┓┃┗━┓
┃┗┛┗┛┃┃┏┓┃┃┏┛┃┏┓┃┃┏━━┛┃┗┛┃┃┏┓┃
┗┓┏┓┏┛┃┗┛┃┃┃━┃┗┛┃┃┗━━┓┃┃┃┃┃┗┛┃
━┗┛┗┛━┗━━┛┗┛━┗━━┛┗━━━┛┗┻┻┛┗━━┛
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
print("""
┏━━━━┓┏━━━┓┃┃┃┃━━┓━━━┓┏━━━┓
┃┏┓┏┓┃┃┏━━┛┃┃┃┃┫┣┛┓┏┓┃┃┏━━┛
┗┛┃┃┗┛┃┗━━┓┃┃┃┃┃┃┃┃┃┃┃┃┗━━┓
┃┃┃┃┃┃┃┏━━┛━━━┓┃┃┃┃┃┃┃┃┏━━┛
┃┏┛┗┓┃┛┗┓┃┃━━━┛┫┣┓┛┗┛┃┛┗┓┃┃
┃┗━━┛┃━━┛┃┃┃┃┃┃━━┛━━━┛━━┛┃┃
┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃
┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃

""")
v = TfidfVectorizer(use_idf=True)
tf_idf_x = v.fit_transform(data)

w2v_x = spacy_word_2_vec(data)

km = KMeans(n_clusters=5, random_state=1)

print("""
 __    __ __       __ ________  ______  __    __  ______  
|  \  /  \  \     /  \        \/      \|  \  |  \/      \ 
| ▓▓ /  ▓▓ ▓▓\   /  ▓▓ ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓\ ▓▓\ | ▓▓  ▓▓▓▓▓▓\\
| ▓▓/  ▓▓| ▓▓▓\ /  ▓▓▓ ▓▓__   | ▓▓__| ▓▓ ▓▓▓\| ▓▓ ▓▓___\▓▓
| ▓▓  ▓▓ | ▓▓▓▓\  ▓▓▓▓ ▓▓  \  | ▓▓    ▓▓ ▓▓▓▓\ ▓▓\▓▓    \ 
| ▓▓▓▓▓\ | ▓▓\▓▓ ▓▓ ▓▓ ▓▓▓▓▓  | ▓▓▓▓▓▓▓▓ ▓▓\▓▓ ▓▓_\▓▓▓▓▓▓\\
| ▓▓ \▓▓\| ▓▓ \▓▓▓| ▓▓ ▓▓_____| ▓▓  | ▓▓ ▓▓ \▓▓▓▓  \__| ▓▓
| ▓▓  \▓▓\ ▓▓  \▓ | ▓▓ ▓▓     \ ▓▓  | ▓▓ ▓▓  \▓▓▓\▓▓    ▓▓
 \▓▓   \▓▓\▓▓      \▓▓\▓▓▓▓▓▓▓▓\▓▓   \▓▓\▓▓   \▓▓ \▓▓▓▓▓▓ 
"""
)

# TF-IDF
tf_idf_model = km.fit(tf_idf_x)
tf_idf_result = tf_idf_model.predict(tf_idf_x)
tf_idf_dict = {}

# Word-to-Vec
w2v_model = km.fit(w2v_x)
w2v_result = w2v_model.predict(w2v_x)
w2v_dict = {}

print("""                                     
▀███▀▀▀██▄ ▀████▀ ▄▄█▀▀▀█▄███▀▀██▀▀███
  ██    ▀██▄ ██ ▄██▀     ▀█▀   ██   ▀█
  ██     ▀██ ██ ██▀       ▀    ██     
  ██      ██ ██ ██             ██     
  ██     ▄██ ██ ██▄            ██     
  ██    ▄██▀ ██ ▀██▄     ▄▀    ██     
▄████████▀ ▄████▄ ▀▀█████▀   ▄████▄   
""")
for i in range(20):
    container = np.random.randint(low=0, high=3944, size=1)
    j = container[0]

    add_values_in_dict(tf_idf_dict, tf_idf_result[j], y[j]+"\n")
    add_values_in_dict(w2v_dict, w2v_result[j], y[j]+"\n")

print("""
   ( .  (   (   (  (   (   (    (( 
  ()) . )\  )\: )\ )\  )\  )\  (\()
 ((_)) ((_)((_)((_)(_)(_()((_)))(_)
(/ __|/ _ \|  \/  | _ \   \ _ \ __|
| (__| (_) | |\/| |  _/ - |   / _| 
 \___|\___/|_|  |_|_| |_|_|_|_\___|

""")
for key in range(0, 5):
    print(key)
    print("TF-IDF cluster\n")
    if key in tf_idf_dict:
        print(*tf_idf_dict[key], sep="\n")
    print("W2V cluster\n")
    if key in w2v_dict:
        print(*w2v_dict[key], sep="\n")



