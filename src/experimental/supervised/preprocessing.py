import pandas as pd
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import spacy
import time


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


def process_stack_trace_row(stack_trace, mode):
    if pd.isnull(stack_trace):
        return []

    # Tokenize Stack Trace
    words = word_tokenize(stack_trace)

    cleaned = clean_text(words)
    filtered = filter_text(cleaned)

    return stem_text(filtered, mode)


def process_stack_trace_column(dataframe, mode):
    dataframe.dropna(inplace=True)
    tokenized = dataframe.iloc[:, -2].apply(word_tokenize)
    cleaned = tokenized.apply(clean_text)
    filtered = cleaned.apply(filter_text)

    dataframe['Stack trace'] = filtered.apply(stem_text, mode=mode)
    return dataframe


def word2vec(dataframe):
    nlp = spacy.load('en_core_web_md')
    docs = dataframe[:].apply(lambda x: nlp(x))
    pdv = []
    for index, value in docs.iteritems():
        pdv.append(value.vector)
    return pdv


def process_stack_trace(dataframe, stem_mode, process_mode):
    start = time.time()

    if process_mode == 'c':
        print(process_stack_trace_column(dataframe, stem_mode))
    else:
        for cols, item in dataframe.iterrows():
            print(process_stack_trace_row(item.iloc[-2], stem_mode))  # Process Stack Trace
            # print(process_stack_trace_row(item['Stack trace'], stem_mode))  # Process Stack Trace

    print("Completed:", time.time() - start)



if __name__ == "__main__":

    df_monkey_labeled = pd.read_csv('../../../data/supervised/monkey_selection.csv')
    df_github_labeled = pd.read_csv('../../../data/supervised/github_selection.csv')

    df = process_stack_trace_column(df_github_labeled, mode='l')



