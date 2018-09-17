# -*- coding: utf8 -*-
from time import time
import os.path
import pandas as pd
import psycopg2
import numpy as np
import re
import string
import sklearn
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
import pickle
from sklearn.externals import joblib

conn = psycopg2.connect(host="10.117.2.34", port="5432", database="overcare", user="overcare", password="overcare")


def preprocessor(df):
    df.columns = ['texto', 'cid']
    # print(df) #imprimindo o dataframe
    # Pegando os textos puros da coluna texto, para normalização.
    # textos puros terão de ter um índice
    df['texto'] = df['texto'].astype(str).str.replace('-', '')
    df['texto'] = df['texto'].astype(str).str.replace('/', '')
    df['texto'] = df['texto'].astype(str).str.replace('+', '')
    df['texto'] = df['texto'].astype(str).str.replace('ões', '')
    df['texto'] = df['texto'].astype(str).str.replace(';', '')
    df['texto'] = df['texto'].astype(str).str.replace('#', '')
    df['texto'] = df['texto'].astype(str).str.replace('~', '')
    df['texto'] = df['texto'].astype(str).str.replace(':', '')
    df['texto'] = df['texto'].astype(str).str.lower().str.split()
    stop = stopwords.words("portuguese")
    textosPuros = df['texto'].apply(lambda x: [w for w in x if not w in stop])
    return textosPuros


def preprocessorNew(textNew):
    # print(df) #imprimindo o dataframe
    words = textNew.lower().split()
    stop = stopwords.words("portuguese")
    # words = ' '.join([w for w in words if not w in stop])
    return words


def dictionary(texto):
    dicionario = set()
    for lista in texto:
        dicionario.update(lista)
    return dicionario


def defineDictionaryPosition(dicionario):
    totalDePalavras = len(dicionario)
    tuplas = zip(dicionario, np.arange(totalDePalavras))
    tradutor = {palavra: indice for palavra, indice in tuplas}
    print("Total de palavras: ")
    print(totalDePalavras)
    return tradutor


def vectorize_text(text, tradutor):
    vector = [0] * len(tradutor)

    for palavra in text:
        if palavra in tradutor:
            position = tradutor[palavra]
            vector[position] += 1

    return vector


def train(texts, tradutor, marcas):
    print("Training.....")
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='portuguese')
    vetoresDeTexto = [vectorize_text(texto, tradutor) for texto in texts]
    # Define o conjunto de dados X
    X = np.array(vetoresDeTexto)
    # Define o conjunto de dados Y (labels)
    Y = np.array(marcas.tolist())
    # Define porcentagem do treino
    porcentagem_de_treino = 0.8
    # Separa o tamanho do treino a partir da porcentagem
    tamanho_do_treino = int(porcentagem_de_treino * len(Y))
    # O restante fica para a validacao
    tamanho_de_validacao = (len(Y) - tamanho_do_treino)
    # Separa os dados de treino
    treino_dados = X[0:tamanho_do_treino]
    # Separa as marcacoes de treino
    treino_marcacoes = Y[0:tamanho_do_treino]
    # Separa os dados de validacao
    validacao_dados = X[tamanho_do_treino:]
    # Separa as marcacoes de validacao
    validacao_marcacoes = Y[tamanho_do_treino:]
    print("Validacao Marcacoes: ")
    print(validacao_marcacoes)
    clf = LogisticRegression()  # MultinomialNB() obtive 62% de acerto#GaussianNB()
    clf.fit(treino_dados, treino_marcacoes)
    # accuracy
    accuracy = clf.score(validacao_dados, validacao_marcacoes)
    file_name = 'train_data.pkl'
    pickle._dump(clf, open(file_name, 'wb'))
    # fit_file = joblib.dump(clf, file_name)
    print("Indice de acerto do algoritmo: ")
    print("%.2f " % round(accuracy * 100) + "%\n")
    print("End of train...")
    predict(file_name)
    # To get a fit_file
    # return fit_file


# just a test
# HOW??
def predict(fit):
    print("\nPredict......")
    # new text to predict
    new = preprocessorNew('new text')
    X = np.array(new)
    # new_text = preprocessorNew(new)
    # To have the fit file \/
    loaded_model = pickle.load(open(fit, 'rb'))
    # how to predict this new data??
    result = loaded_model.predict(X.reshape(1, -1))
    print(result)


df1 = pd.read_sql(
    "SELECT dsobservacaoclinica1, cdcidcategoria AS CID FROM iaconsultas limit 500",
    conn)
df1.columns = ['texto', 'cid']
marca = df1['cid']
textosPuros = preprocessor(df1)
print("Texts...")
print(textosPuros)
dict = dictionary(textosPuros)
print("Dict....")
print(dict)
translate = defineDictionaryPosition(dict)
# to train
train(textosPuros, translate, marca)
# TEST
# predict(textosPuros, translate, marca)
