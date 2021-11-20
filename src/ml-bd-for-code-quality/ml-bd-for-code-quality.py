import pandas as pd
import re
from string import punctuation

# Pre Processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Word Embedding
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
    stop_words = set(stopwords.words('english'))
    r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))

    dataframe.dropna(inplace=True)
    tokenized = dataframe.iloc[:, -1].apply(word_tokenize)
    cleaned = tokenized.apply(clean_text)
    filtered = cleaned.apply(filter_text)

    return filtered.apply(stem_text, mode=mode)


def process_stack_trace(dataframe, stem_mode, process_mode):
    start = time.time()

    if process_mode == 'c':
        to_vec_dataframe = pd.DataFrame(process_stack_trace_column(dataframe, stem_mode))
        # print(to_vec_dataframe)
        sklearn_vector(to_vec_dataframe)
    else:
        for cols, item in dataframe.iterrows():
            print(process_stack_trace_row(item.iloc[-1], stem_mode))  # Process Stack Trace
            # print(process_stack_trace_row(item['Stack trace'], stem_mode))  # Process Stack Trace

    print("Completed:", time.time() - start)


def sklearn_vector(dataframe):
    print("V1")
    for i in dataframe.iterrows():
        print(i)
    # print(dataframe)
    # tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    # tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(dataframe)
    print("V2")
    # cv = CountVectorizer()
    # word_count_vector = cv.fit_transform(dataframe)
    # print("Vector:", word_count_vector.shape)

    # tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    # tfidf_transformer.fit(word_count_vector)
    #
    # df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names_out(), columns=["idf_weights"])
    # print(df_idf.sort_values(by=['idf_weights']))


if __name__ == "__main__":

    # df_github_own = pd.read_csv('full_github_issues.csv')
    # df_monkey_own = pd.read_csv('full_monkey.csv')

    df_monkey = pd.read_csv('../../data/monkey_data_stack_trace.csv')
    df_github = pd.read_csv('../../data/github_issues_stack_trace.csv')
    df_w3c = pd.read_csv('../../data/w3c_test_results_failed.csv')

    process_stack_trace(df_monkey, stem_mode='l', process_mode='c')
