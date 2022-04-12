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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

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

    return filtered.apply(stem_text, mode=mode)


def process_stack_trace(dataframe, stem_mode, process_mode):
    start = time.time()

    if process_mode == 'c':
        to_vec_dataframe = pd.DataFrame(process_stack_trace_column(dataframe, stem_mode))
        # print(to_vec_dataframe)
        sklearn_vector(to_vec_dataframe)
    else:
        for cols, item in dataframe.iterrows():
            # print(process_stack_trace_row(item.iloc[-1], stem_mode))  # Process Stack Trace
            # print(process_stack_trace_row(item['Stack trace'], stem_mode))  # Process Stack Trace
            sklearn_vector(process_stack_trace_row(item.iloc[-1], stem_mode))

    print("Completed:", time.time() - start)


def sklearn_vector(dataframe):
    # change the form from lists to Dataframe with strings
    dfa = []
    # apply instead of for loop in PreProcessing
    for i in dataframe['Stack trace']:
        string = ' '.join(i)
        dfa.append(string)
        # print(str)
    # for i in dataframe['Stack trace']:
    #     dfa.append(' '.join(i))

    dfx = pd.DataFrame(dfa, columns=['StackTrace'])
    # print(dfx)

    # call function to vectorize
    spacy_word_2_vec(dataframe)
    # sklearn_vector_vectorizer(dfx)
    # sklearn_vector_transformer(dfx)


def spacy_word_2_vec(dataframe):
    # print(dataframe)
    nlp = spacy.load('en_core_web_md')

    # wec = []
    # dataframe["StackTrace"].apply(lambda row: wec.append(nlp(row)))
    # print(wec)

    print("NLP")
    docs = list(nlp.pipe(dataframe['StackTrace'], n_process=8))
    [print(i.vector) for i in docs]


def sklearn_vector_vectorizer(dfx):
    v = TfidfVectorizer()
    x = v.fit_transform(dfx['StackTrace'])

    # print(x)
    df = pd.DataFrame(x.toarray(), columns=v.get_feature_names_out())
    print(df)


def sklearn_vector_transformer(dfx):
    # # create the vector
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(dfx['StackTrace'])
    print("Vector shape", word_count_vector.shape)

    # calculate the idf values
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names_out(), columns=["idf_weights"])

    # sort ascending by idf values
    df_idf_sorted = df_idf.sort_values(by=['idf_weights'])

    # print short versions
    # print(df_idf)
    # print(df_idf_sorted)

    # print the whole dataframe
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df_idf_sorted)
    #     print(df_idf)

    # compute TFIDF score
    # count matrix
    count_vector = cv.transform(dfx['StackTrace'])
    tf_idf_vector = tfidf_transformer.transform(count_vector)
    feature_names = cv.get_feature_names_out()
    first_document_vector = tf_idf_vector[0]
    # print(tf_idf_vector)
    dff = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,):
        print(dff.sort_values(by=["tfidf"], ascending=True))
        print(dfx.iloc[0])


if __name__ == "__main__":

    # df_github_own = pd.read_csv('../../data/full_github_issues.csv')
    # df_monkey_own = pd.read_csv('../../data/full_monkey.csv')

    df_monkey = pd.read_csv('../../../data/unsupervised/monkey_data_stack_trace.csv')
    df_github = pd.read_csv('../../../data/unsupervised/github_issues_stack_trace.csv')
    df_w3c = pd.read_csv('../../../data/unsupervised/w3c_test_results_failed.csv')

    process_stack_trace(df_monkey, stem_mode='l', process_mode='c')
