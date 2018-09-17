# -*- coding: utf-8 -*-

import os.path
import logging
import argparse
import psycopg2
import unicodedata

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords

import sklearn
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


def train(df, fit_file):
    print("\nTraining...")
    df = df.dropna()
    train_size = 0.8
    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None
    )
    logreg = LogisticRegression()
    pipe = Pipeline([('vect', vectorizer), ('logreg', logreg)])
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        df.texto, df.categoria, train_size=train_size
    )
    pipe.fit(X_train, Y_train)
    accuracy = pipe.score(X_test, Y_test)
    msg = "\nAccuracy with {:.0%} of testing data: {:.1%}\n".format(1 - train_size, accuracy)
    print(msg)
    pipe.fit(df.texto, df.categoria)
    joblib.dump(pipe, fit_file)


def predict(url, fit_file):
    pipe = joblib.load(fit_file)
    words = pre_processor(url)
    resp = pipe.predict([words])
    print("\nCategory: %s \n" % resp[0])
    resp = zip(pipe.classes_, pipe.predict_proba([words])[0])
    resp.sort(key=lambda tup: tup[1], reverse=True)
    for cat, prob in resp:
        print("Category {:16s} with {:.1%} probab.".format(cat, prob))


def to_unicode(data):
    try:
        data = data.decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        try:
            data = data.decode('iso-8859-1')
        except (UnicodeDecodeError, UnicodeEncodeError):
            try:
                data = data.decode('latin-1')
            except (UnicodeDecodeError, UnicodeEncodeError):
                data = data
    return data


'''
def remove_nonlatin(string):
    new_chars = []
    for char in string:
        if char == '\n':
            new_chars.append(' ')
            continue
        try:
            if unicodedata.name(unicode(char)).startswith(('LATIN', 'SPACE')):
                new_chars.append(char)
        except:
            continue
    return ''.join(new_chars)
'''


def pre_processor(link):
    stops = set(stopwords.words("portuguese"))
    #goose = Goose()
    article = link
    text = article.cleaned_text
    # text = remove_nonlatin(to_unicode(text))
    words = text.lower().split()
    words = ' '.join([w for w in words if not w in stops])
    return words


def worker(link, categ, lines):
    words = pre_processor(link)
    print
    "{:6d} words in: \t {:.70}".format(len(words), link)
    lines.append((link, categ, words))


def main(links_file, fit_file, to_predict):
    if not os.path.isfile(fit_file):
        df = prepare_data(links_file)
        train(df, fit_file)

    if to_predict:
        predict(to_predict, fit_file)


def prepare_data(links_file):
    pool = ThreadPoolExecutor(8)
    conn = psycopg2.connect(host="10.117.2.34", port="5432", database="overcare", user="overcare", password="overcare")

    # dataframe com o texto dos sintomas e a categoria do cid
    df = pd.read_sql(
        "SELECT dsobservacaoclinica1, cdcidcategoria AS CID FROM iaconsultas limit 500",
        conn)
    # criação das colunas do dataframe
    lines = []

    if links_file:
        links = pd.read_csv(links_file, sep=';')
        links = ((r['link'], r['categ']) for i, r in links.iterrows())

    print
    "Downloading and processing data...\n"
    for link, categ in links:
        pool.submit(worker, link, categ, lines)

    pool.shutdown(wait=True)
    df = DataFrame(lines)
    df.columns = ['link', 'categoria', 'texto']
    df.to_csv('bag_words.csv', sep=';', encoding='utf-8')
    print("\nSaving bag_words.csv to future use.")
    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='')

    parser.add_argument(
        '-t', '--train',
        help='Train and save to file',
    )

    parser.add_argument(
        '-p', '--predict',
        help='Predict',
    )

    parser.add_argument(
        '-f', '--file',
        help='News links file',
    )

    args = parser.parse_args()
    links_file = args.file
    to_predict = args.predict
    fit_file = args.train

    try:
        main(links_file, fit_file, to_predict)
    except Exception as e:
        logging.exception(e)
