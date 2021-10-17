import pandas as pd
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

def get_data():
    # GITHUB-ISSUES
    # github_issues_url = "https://github.com/tingsu/DroidDefects/blob/master/ground-truth-cases/Dataset_crashanalysis/Github_issues.csv?raw=true"
    # df_github = pd.read_csv(github_issues_url)
    # print(df_github[['Pkg name', 'Exception name']])

    # MONKEY_DATA
    monkey_data_url = "https://github.com/tingsu/DroidDefects/blob/master/ground-truth-cases/Dataset_crashanalysis" \
                      "/Monkey_data.csv?raw=true "
    df_monkey = pd.read_csv(monkey_data_url)

    for col, item in df_monkey.iterrows():
        folder = item[0]
        bug_report = item[4]
        file = f"https://github.com/tingsu/DroidDefects/blob/master/ground-truth-cases/Dataset_crashanalysis/Monkey_data/{folder}/unique/{bug_report}"

        # Problem: StackTrace not directly in CSV, but only in sub-folders of repository

        print(file)


def process_text():
    stack_trace = "java.lang.RuntimeException: Unable to resume activity {net.etuldan.sparss/net.etuldan.sparss.activity.HomeActivity}: java.lang.NullPointerException: Attempt to invoke virtual method 'void net.etuldan.sparss.adapter.DrawerAdapter.a(int)' on a null object reference"
    words = word_tokenize(stack_trace)
    stop_words = set(stopwords.words('english'))

    cleaned = []
    for w in words:
        strings = re.split(r"[.|,'/]", w)
        for s in strings:
            cleaned.append(s)

    filtered = [w.lower() for w in cleaned if w not in stop_words and w not in set(punctuation) and w != '']

    l = WordNetLemmatizer()
    lemmatized = [l.lemmatize(w) for w in filtered]

    s = PorterStemmer()
    stemmed = [s.stem(w) for w in filtered]

    print("Lemmatized:", lemmatized)
    print("Stemmed:", stemmed)


if __name__ == "__main__":
    # get_data()
    process_text()