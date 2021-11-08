import pandas as pd
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import time
import requests  # Used to get content of github files


def get_monkey_data():
    # READ MONKEY_DATA + ADD STACK TRACES
    monkey_data_url = "https://github.com/tingsu/DroidDefects/blob/master/ground-truth-cases/Dataset_crashanalysis" \
                      "/Monkey_data.csv?raw=true "
    df_monkey = pd.read_csv(monkey_data_url)

    monkey_stack_traces = []

    counter = 0
    s = requests.Session()
    for col, item in df_monkey.iterrows():
        folder = item[0]
        bug_report = item['Stack Trace']

        url = f"https://github.com/tingsu/DroidDefects/blob/master/ground-truth-cases/Dataset_crashanalysis/Monkey_data/{folder}/unique/{bug_report}?raw=true"
        get_stack_traces(s, monkey_stack_traces, url)

        counter += 1
        print(counter)

    df_stack_traces = pd.DataFrame({'Stack Trace': monkey_stack_traces})
    df_monkey = df_monkey.join(df_stack_traces)

    df_monkey.to_csv(path_or_buf='full_monkey_new.csv')

    return df_monkey


def get_github_data():
    # GITHUB-ISSUES
    github_issues_url = "https://github.com/tingsu/DroidDefects/blob/master/ground-truth-cases/Dataset_crashanalysis/Github_issues.csv?raw=true"
    df_github = pd.read_csv(github_issues_url)

    github_stack_traces = []

    s = requests.Session()
    counter = 0
    for col, item in df_github.iterrows():
        folder = item['Project'].replace('/', '_')
        bug_report = item['Issue ID']

        url = f"https://raw.githubusercontent.com/tingsu/DroidDefects/master/ground-truth-cases/Dataset_crashanalysis/Github_issues/{folder}/{bug_report}.txt"
        get_stack_traces(s, github_stack_traces, url)

        counter += 1
        print(counter)

    df_stack_traces = pd.DataFrame({'Stack Trace': github_stack_traces})
    df_github = df_github.join(df_stack_traces)

    df_github.to_csv(path_or_buf='full_github_issues.csv')

    return df_github


def get_stack_traces(session, df, url):
    stack_trace = session.get(url)
    text_content = stack_trace.text.replace('\n', '').replace('\t', '')
    df.append(text_content)
    return df


def process_text(file, mode):
    if pd.isnull(file):
        return []

    words = word_tokenize(file)
    stop_words = set(stopwords.words('english'))

    # Split words at punctuation
    r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))
    cleaned = []
    for w in words:
        strings = r.split(w)
        for s in strings:
            cleaned.append(s)

    filtered = [w.lower() for w in cleaned          # Convert to Lowercase
                if w not in stop_words              # Remove Stop words
                and w not in set(punctuation)       # Remove Special Characters
                and not w.isdigit()                 # Remove Numbers
                and w != '']                        # Remove Whitespaces

    if mode == 'l':
        l = WordNetLemmatizer()
        lemm = [l.lemmatize(w) for w in filtered]
        return lemm
    elif mode == 's':
        s = PorterStemmer()
        stem = [s.stem(w) for w in filtered]
        return stem


def process_stack_trace(dataframe, mode):
    start = time.time()

    for cols, item in dataframe.iterrows():
        print(item[0])  # Print Item ID
        print(process_text(item.iloc[-1], mode))  # Process Stack Trace
        # print(process_text(item.['Stack trace'], mode))  # Process Stack Trace

    print("Completed:", time.time() - start)

# Too Slow
#########################
#def process_stack_trace_by_preprocess(dataframe, mode):
#     words = []
#     cleaned = []
#     filtered = []
#     stop_words = set(stopwords.words('english'))
#
#     for cols, item in dataframe.iterrows():
#         words_item = word_tokenize(item.iloc[-1])
#         words.append(words_item)
#
#     r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))
#     cleaned_item = []
#     for word_item in words:
#         for w in word_item:
#             strings = r.split(w)
#             for s in strings:
#                 cleaned_item.append(s)
#             cleaned.append(cleaned_item)
#
#     for cleaned_item in cleaned:
#         filtered_item = []
#         filtered_item = [w.lower() for w in cleaned_item  # Convert to Lowercase
#                          if w not in stop_words  # Remove Stop words
#                          and w not in set(punctuation)  # Remove Special Characters
#                          and not w.isdigit()  # Remove Numbers
#                          and w != '']  # Remove Whitespaces
#         filtered.append(filtered_item)
#
#     for filtered_item in filtered:
#         if mode == 'l':
#             l = WordNetLemmatizer()
#             lemm = [l.lemmatize(w) for w in filtered_item]
#             print(lemm)
#         elif mode == 's':
#             s = PorterStemmer()
#             stem = [s.stem(w) for w in filtered_item]
#             print(stem)


if __name__ == "__main__":

    df_github_own = pd.read_csv('full_github_issues.csv')
    df_monkey_own = pd.read_csv('full_monkey.csv')

    df_monkey = pd.read_csv('../../data/monkey_data_stack_trace.csv')
    df_github = pd.read_csv('../../data/github_issues_stack_trace.csv')
    df_w3c = pd.read_csv('../../data/w3c_test_results_failed.csv')

    process_stack_trace(df_monkey, 'l')

    #process_stack_trace_by_preprocess(df, 's')

    # Timing Monkey Data (monkey_data_stack_trace.csv):
    # ---- Stemming ----
    # * 1. 24.807677030563354
    # * 2. 24.937902450561523
    # * 3.
    # * 4.
    # ---- Lemmatizing ----
    # * 1. 16.52564287185669
    # * 2. 16.96170926094055
    # * 3.
    # * 4.

    # Timing Github Issues (github_issues_stack_trace):
    # ---- Stemming ----
    # * 1. 47.79965376853943
    # * 2. 45.37876486778259
    # * 3.
    # * 4.
    # ---- Lemmatizing ----
    # * 1. 31.144325017929077
    # * 2. 31.442722082138066
    # * 3.
    # * 4.

    # Join csv with corresponding stack traces
    # df = get_github_data()
    # df = get_monkey_data()


