import pandas as pd
import re
from string import punctuation

# Pre Processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Word Embedding
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Others
import time


def clean_text(words):
    r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))
    cleaned = []
    for w in words:
        strings = r.split(w)
        for s in strings:
            cleaned.append(s)

    return cleaned


def filter_text(words):
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
        return lemm
    elif mode == 's':
        s = PorterStemmer()
        stem = [s.stem(w) for w in words]
        return stem


def process_stack_trace_row(stack_trace, mode):
    if pd.isnull(stack_trace):
        return []

    # Tokenize Stack Trace
    words = word_tokenize(stack_trace)
    # Split words at punctuation
    cleaned = clean_text(words)
    # Filter out unnecessary characters
    filtered = filter_text(cleaned)

    return stem_text(filtered, mode)


def process_stack_trace_column(dataframe, mode):
    dataframe.dropna(inplace=True)
    tokenized = dataframe.iloc[:, -1].apply(word_tokenize)
    cleaned = tokenized.apply(clean_text)
    filtered = cleaned.apply(filter_text)
    dfa = []

    dataframe = filtered.apply(stem_text, mode=mode)
    dataframe.apply(lambda row: dfa.append(' '.join(row)))
    dfx = pd.DataFrame(dfa, columns=['StackTrace'])

    return dfx


def process_stack_trace(dataframe, stem_mode, process_mode, vector_mode):
    start = time.time()

    if process_mode == 'c':
        to_vec_dataframe = process_stack_trace_column(dataframe, stem_mode)
        sklearn_vector(to_vec_dataframe, vector_mode)
    else:
        for cols, item in dataframe.iterrows():
            print(process_stack_trace_row(item.iloc[-1], stem_mode))  # Process Stack Trace
            # print(process_stack_trace_row(item['Stack trace'], stem_mode))  # Process Stack Trace
            # sklearn_vector(process_stack_trace_row(item.iloc[-1], stem_mode))

    print("Completed:", time.time() - start)


def sklearn_vector(dataframe, vector_mode):
    # call function to vectorize
    if vector_mode == 'wv':
        spacy_word_2_vec(dataframe)
    elif vector_mode == 'sv':
        sklearn_vector_vectorizer(dataframe)
    else:
        print("Wrong vector_mode!\nUse:\n  'sv'\n  'wv'")


def spacy_word_2_vec(dataframe):
    # print(dataframe)
    nlp = spacy.load('en_core_web_md')
    # loop every Document and concate to dp.Dataframe

    # wec = nlp(dataframe)
    # print(wec.vector)


def sklearn_vector_vectorizer(dataframe):
    # print(dataframe)
    v = TfidfVectorizer()
    x = v.fit_transform(dataframe['StackTrace'])
    # print(x)
    df = pd.DataFrame(x.toarray(), columns=v.get_feature_names_out())
    print(df)


if __name__ == "__main__":

    # df_github_own = pd.read_csv('full_github_issues.csv')
    # df_monkey_own = pd.read_csv('full_monkey.csv')

    df_monkey = pd.read_csv('../../data/monkey_data_stack_trace.csv')
    df_github = pd.read_csv('../../data/github_issues_stack_trace.csv')
    df_w3c = pd.read_csv('../../data/w3c_test_results_failed.csv')

    process_stack_trace(df_monkey, stem_mode='l', process_mode='c', vector_mode='sv')
    # Modes:
    # stem_mode = 'l' || 's'
    # process_mode = 'c' || 'r' //'r' is not suitable for vector_mode
    # vector_mode = 'sv' || 'wv'
