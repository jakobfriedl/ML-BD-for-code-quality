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
    df.append(stack_trace.text)
    return df


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

    cleaned = clean_text(words)
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
        print(process_stack_trace_column(dataframe, stem_mode))
    else:
        for cols, item in dataframe.iterrows():
            print(process_stack_trace_row(item.iloc[-1], stem_mode))  # Process Stack Trace
            # print(process_stack_trace_row(item['Stack trace'], stem_mode))  # Process Stack Trace

    print("Completed:", time.time() - start)


if __name__ == "__main__":

    df_github_own = pd.read_csv('full_github_issues.csv')
    df_monkey_own = pd.read_csv('full_monkey.csv')

    df_monkey = pd.read_csv('../../data/monkey_data_stack_trace.csv')
    df_github = pd.read_csv('../../data/github_issues_stack_trace.csv')
    df_w3c = pd.read_csv('../../data/w3c_test_results_failed.csv')

    process_stack_trace(df_monkey, stem_mode='l', process_mode='c')

    # Timing Monkey Data (monkey_data_stack_trace.csv) :
    # ---- Stemming ----
    # BY ROW:
    #   * 1. 27.73685073852539
    #   * 2. 25.155765295028687
    # BY COL:
    #   * 1. 22.720109701156616
    #   * 2. 22.477178812026978
    # ---- Lemmatizing ----
    # BY ROW:
    #   * 1. 17.941423892974854
    #   * 2. 16.916264295578003
    # BY COL:
    #   * 1. 14.2380051612854
    #   * 2. 14.486801385879517

    # Timing Github Issues (github_issues_stack_trace):
    # ---- Stemming ----
    # BY ROW:
    #   * 1. 52.6889374256134
    #   * 2. 47.71541500091553
    # BY COL:
    #   * 1. 33.59704065322876
    #   * 2. 35.45515513420105
    # ---- Lemmatizing ----
    # BY ROW:
    #   * 1. 33.44994807243347
    #   * 2. 33.6430242061615
    # BY COL:
    #   * 1. 21.004103183746338
    #   * 2. 21.5268771648407

    # Join csv with corresponding stack traces
    # df = get_github_data()
    # df = get_monkey_data()


