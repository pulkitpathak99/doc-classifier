import pickle
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords
import pandas as pd
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# stopwords.words('english')
# string.punctuation
ps = PorterStemmer()
# ps.stem('worries')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


def preprocesse_df(df):
    df = df.dropna()
    df.duplicated().sum()
    df = df.drop_duplicates(keep='first')
    df['transformed_content'] = df['content'].apply(transform_text)
    processed_df = df[df['content'].str.len() >= 1000]

    return processed_df
