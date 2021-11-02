import pandas as pd
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import requests


def get_data():
    # GITHUB-ISSUES
    # github_issues_url = "https://github.com/tingsu/DroidDefects/blob/master/ground-truth-cases/Dataset_crashanalysis/Github_issues.csv?raw=true"
    # df_github = pd.read_csv(github_issues_url)
    # print(df_github[['Pkg name', 'Exception name']])

    # READ MONKEY_DATA + ADD STACK TRACES
    # monkey_data_url = "https://github.com/tingsu/DroidDefects/blob/master/ground-truth-cases/Dataset_crashanalysis" \
    #                   "/Monkey_data.csv?raw=true "
    # df_monkey = pd.read_csv(monkey_data_url)
    # monkey_stack_traces = []
    # counter = 0
    # s = requests.Session()
    # for col, item in df_monkey.iterrows():
    #     folder = item[0]
    #     bug_report = item[4]
    #     url = f"https://github.com/tingsu/DroidDefects/blob/master/ground-truth-cases/Dataset_crashanalysis/Monkey_data/{folder}/unique/{bug_report}?raw=true"
    #
    #     get_stack_traces(s, monkey_stack_traces, url)
    #
    #     counter += 1
    #     print(counter)
    #
    # df_stack_traces = pd.DataFrame({'Stack Trace': monkey_stack_traces})
    #
    # df_monkey = df_monkey.join(df_stack_traces)
    #
    # print(df_monkey.head(5))
    #
    # df_monkey.to_csv(path_or_buf='full_monkey.csv')

    test = pd.read_csv('full_monkey.csv')
    print(test)

def get_stack_traces(session, df, url):
    stack_trace = session.get(url)
    df.append(stack_trace.text.replace('\n', ' '))
    return df

def process_text(file):
    words = word_tokenize(file)
    stop_words = set(stopwords.words('english'))

    # Split words at punctuation
    r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))
    cleaned = []
    for w in words:
        strings = r.split(w)
        for s in strings:
            cleaned.append(s)

    filtered = [w.lower() for w in cleaned if w not in stop_words and w not in set(punctuation) and w != '']

    l = WordNetLemmatizer()
    lemm = [l.lemmatize(w) for w in filtered]
    print(f"Lemmatized: {lemm}")

    s = PorterStemmer()
    stemm = [s.stem(w) for w in filtered]
    print(f"Stemmed: {stemm}")



if __name__ == "__main__":
    get_data()
    # example_file = "java.lang.RuntimeException: Unable to resume activity {net.etuldan.sparss/net.etuldan.sparss.activity.HomeActivity}: java.lang.NullPointerException: Attempt to invoke virtual method 'void net.etuldan.sparss.adapter.DrawerAdapter.a(int)' on a null object reference"
    # print(f"Original file {example_file}\n")
    # process_text(example_file)
