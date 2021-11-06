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

    if mode == 's':
        s = PorterStemmer()
        stem = [s.stem(w) for w in filtered]
        return stem


def process_stack_trace(dataframe, mode):
    start = time.time()

    for cols, item in dataframe.iterrows():
        print(item[0])  # Print Item ID

        print(process_text(item['Stack Trace'], mode))  # Process Text

    print("Completed:", time.time() - start)


if __name__ == "__main__":

    # df = pd.read_csv('full_github_issues.csv')
    df = pd.read_csv('full_monkey.csv')

    process_stack_trace(df, 's')

    # Timing Monkey Data:
    # ---- Stemming ----
    # * 1. 51.293915033340454
    # * 2. 52.06282424926758
    # * 3. 46.19591021537781
    # ---- Lemmatizing ----
    # * 1. 37.13634729385376
    # * 2. 37.823638677597046
    # * 3. 39.39327526092529


    # df = get_github_data()
    # df = get_monkey_data()

    # print(df)
    # example_stack_trace = df.iat[0, 8]  # Get StackTrace of first Entry of Dataframe
    # print(f"Original: {example_stack_trace}\n")
    #
    # # Start timer
    # start = time.time()
    # process_text(example_stack_trace)
    # # End timer
    # print(f"Finished Text-Processing in {time.time()-start} seconds")

    # only Lemmatization: 0.9846229553222656 seconds
    # only Stemming: 0.015641450881958008 seconds
    # both: 1.0480988025665283 seconds


