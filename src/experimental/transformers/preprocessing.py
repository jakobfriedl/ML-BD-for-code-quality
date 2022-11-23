import pandas as pd
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from experimental.transformers.special_characters import characters
import time

def replace_special_characters(words):
    return [characters[w] if w in characters else w for w in words]

def clean_text(words):
    # Split words at punctuation
    r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))
    cleaned = []
    for w in words:
        strings = r.split(w)
        for s in strings:
            cleaned.append(s)

    return cleaned


def filter_text(words):
    # Filter out unnecessary characters
    stop_words = set(stopwords.words('english'))

    return [w.lower() for w in words            # Convert to Lowercase
            if w not in stop_words              # Remove Stop words
            and w not in set(punctuation)       # Remove Special Characters
            and not w.isdigit()                 # Remove Numbers
            and w != '']                        # Remove Whitespaces


def stem_text(words, mode):
    if mode == 'l':
        l = WordNetLemmatizer()
        lemm = [l.lemmatize(w) for w in words]
        return ' '.join(x for x in lemm if x != '\n')
    elif mode == 's':
        s = PorterStemmer()
        stem = [s.stem(w) for w in words]
        return ' '.join(x for x in stem if x != '\n')
    else:
        print("Invalid mode.")
        return


def process_stack_trace_column(dataframe, data_col, replace_special_chars=False, stem_mode='l'):
    print("Preprocessing started")
    dataframe.dropna(inplace=True)
    tokenized = dataframe[data_col].apply(word_tokenize)
    if replace_special_chars:
        replaced = tokenized.apply(replace_special_characters)
        cleaned = replaced.apply(clean_text)
    else:
        cleaned = tokenized.apply(clean_text)
    filtered = cleaned.apply(filter_text)

    dataframe[data_col] = filtered.apply(stem_text, mode=stem_mode)
    print("Preprocessing finished")
    return dataframe


language_dataset = "../../../data/language_data/language_dataset.csv"
out_file = "../../../data/language_data/v2_processed_language_dataset_with_special_chars.csv"

start = time.time()
df_languages = pd.read_csv(language_dataset)
df = process_stack_trace_column(df_languages, "Code", replace_special_chars=True, stem_mode='l')

df.to_csv(out_file, index=False)

print(time.time() - start)