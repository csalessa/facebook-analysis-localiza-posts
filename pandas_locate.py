import pandas as pd
import re
import csv,os,sys
import pprint
import ftfy
from unidecode import unidecode
import time
import datetime


BASE_DATA_FRAME = pd.read_csv("files/FacebookLocaliza.csv", encoding="utf-8")
RANKED_DATA_FRAME = pd.read_csv("files/ranked_words.csv", encoding="utf-8")
RESULT_FRAME = pd.DataFrame([], columns=('word', 'occurrence'))
UPDATE_RANKED_WORDS = False
MIMIMUN_DATE = "01/03/2019"
PANDEMIC_TRESHOLD_DATE = "15/03/2020"


def main():
    print("Data processed before pandemic")
    data_pre = process_data(filter_pre_pandemic(BASE_DATA_FRAME))
    print("Data processed after pandemic")
    data_pos = process_data(filter_pos_pandemic(BASE_DATA_FRAME))
    data_pre.to_csv("files/data_pre.csv")
    data_pos.to_csv("files/data_pos.csv")

def process_data(df):
    RESULT_FRAME = get_ranking_of_words(sanitize_frame(df))
    if UPDATE_RANKED_WORDS: update_ranked_words()
    RESULT_FRAME = remove_undesired_words(RESULT_FRAME)
    print("df sorted by likes:")
    pprint.pprint(
        RESULT_FRAME.sort_values("likes", ascending=False)
    )
    return RESULT_FRAME.sort_values("likes", ascending=False)

def filter_pre_pandemic(df):
    df.time = pd.to_datetime(df.time)
    df = df[df.time >  pd.to_datetime(MIMIMUN_DATE, dayfirst=True)]
    df = df[df.time < pd.to_datetime(PANDEMIC_TRESHOLD_DATE, dayfirst=True)]
    return df

def filter_pos_pandemic(df):
    df.time = pd.to_datetime(df.time)
    df = df[df.time > pd.to_datetime(PANDEMIC_TRESHOLD_DATE, dayfirst=True)]
    return df

def update_ranked_words():
    RESULT_FRAME.sort_values("occurrence", ascending=False).to_csv("files/ranked_words.csv")

def read_text():
    return str(DATA_FRAME.text.values)

def read_splited_text():
    return str(DATA_FRAME.splited_text.values)

def show_head():
    return DATA_FRAME.head()

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               u"\u0022"
                               '\n'
                               '\\n'
                               "\-"
                               "\;"
                               '<3' '.' ',' '!' '?' '-'
                               "]+", flags=re.UNICODE)
    emoji_pattern.sub(r'', string)
    string = unidecode(ftfy.fix_text(string))
    return emoji_pattern.sub(r'', string)

def sanitize_frame(df):
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].apply(lambda x: remove_emoji(x))
    df["translatedText"] = df["translatedText"].astype(str)
    df["translatedText"] = df["translatedText"].apply(lambda x: remove_emoji(x))
    return df

def remove_undesired_words(df):
    black_list = [
        "a", "e", "i", "o", "u",
        "as", "es", "is", "os", "us",
        "da", "de", "di", "do", "du",
        " ", "  ", "", "com", "pra", "para",
        "na", "no", "que", "um", "em", "mais", "uns",
        "e", "de", "a", "o", "com", "que", "do", "para", "da", "no", "um",
        "mais", "em", "uma", "na", "E",
        "por", "pra", "as", "se", "seu", "sua",
        ">>", ".",
        "esse", "essa", "pelo", "tem", "A",
        "ao", "so", "dos", "esta", "ou",
        "ate", "sobre", "ser", "ter", "tem", "são", "NaN",
        "não", "sim", "nao",
    ]
    df = df.dropna()
    return df[~df["word"].isin(black_list)]

def get_ranking_of_words(DATA_FRAME):
    # Splits phrases into array of words
    df = DATA_FRAME

    # Deprecated
    # iterate_data_frame(df)

    # Remove emoticons and unspected characters
    df = sanitize_frame(df)
    # Split text column into a new data frame then join it with df
    df = df.join(
                df["text"].str.split(" ", expand=True).stack()
                .reset_index(drop=True, level=1)
                .rename("word")
                )

    # cast all words to lowercase
    df["word"] = df["word"].str.lower()

    # filter columns
    df = df[["comments", "likes", "text", "word", "time", "timestamp"]]

    # get series of occurrences for each word, then turn series into frame
    occurrences = df["word"].value_counts().rename("occurrences").to_frame()

    # merge df with occurrences using "word" for left key and "index" for right key
    df = pd.merge(df, occurrences, left_on="word", right_index=True)

    # eliminate word duplicates, sum comments and likes,
    # keep occurrences as all the values are the same
    df = df.groupby("word", as_index=False)\
        .agg({"comments":"sum", "likes":"sum", "occurrences": "max"})

    return df

@DeprecationWarning
def iterate_data_frame(df, rf):
    df['splited_text'] = df["text"].str.split(" ")
    i = 0
    # iterate every row (array of words)
    # TODO: use dataframes instead of these slow for's
    for post in df['splited_text']:
        print("processing post " + str(i) + " of " + str(df['splited_text'].count()))
        i += 1
        # iterate each word
        for word in post:
            # check if the word is not in the RESULT_FRAME
            # then add the new word in RESULT_FRAME
            if rf[rf.word == word].size == 0:
                new_line = pd.DataFrame(
                    {
                        "word": [word],
                        "occurrence": 1
                    }, index=[1]
                )
                # Add the word to RESULT_FRAME
                rf = pd.concat([rf, new_line])
            else:
                # increment by one every column word = var word.
                # note: there should be only one word of a kind here
                rf.loc[rf.word == word, "occurrence"] += 1
    return rf

main()
