# -*- coding: utf8 -*-
from time import time

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

# dataframe com o texto dos sintomas e a categoria do cid
df = pd.read_sql(
    "SELECT dsobservacaoclinica1, cdcidcategoria AS CID FROM iaconsultas limit 500",
    conn)
# criação das colunas do dataframe
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


# textosPuros = df['texto'].astype(str).str.replace('\n', '')

# print(textosPuros)

def remove(string):
    novo = []
    for x in string:
        item = x
        for y in ['\n', '\t', '/', '.', '-', '(', ')']:
            item = item.replace(y, "")
        novo.append(item)
    return novo


# print(textosPuros)

textoMinusculo = textosPuros

# print(textoMinusculo)

textoLimpo = textoMinusculo  # [item for item in textoMinusculo if item not in ['\n', '\t']]

# textoLimpo = re.sub()
dicionario = set()
for lista in textoLimpo:
    dicionario.update(lista)
# imprime dicionario
print("Dicioonario")
# dicionario = {y.strip('\t\n.,/\1234567890();:-_') for y in dicionario}
print(dicionario)
# imprime as palavras limpas
print(textoLimpo)

# Atribui cada palavra a uma posicao no dicionario
totalDePalavras = len(dicionario)
tuplas = zip(dicionario, np.arange(totalDePalavras))
tradutor = {palavra: indice for palavra, indice in tuplas}

# Mostra a quantidade total de palavras
print("Total de palavras: ")
print(totalDePalavras)


def vetorizar_texto(texto, tradutor):
    vetor = [0] * len(tradutor)

    for palavra in texto:
        if palavra in tradutor:
            posicao = tradutor[palavra]
            vetor[posicao] += 1

    return vetor


# Vincula os textos quebrados a posicao no vetor
vetoresDeTexto = [vetorizar_texto(texto, tradutor) for texto in textoLimpo]
marcas = df['cid']
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

print("Frases disponiveis: ")
# print(len(Y))
print("Frases para treino: ")
# print(tamanho_do_treino)
print("Frase para validacao: ")
# print(tamanho_de_validacao)

# Separa os dados de treino
treino_dados = X[0:tamanho_do_treino]
# Separa as marcacoes de treino
treino_marcacoes = Y[0:tamanho_do_treino]
# Separa os dados de validacao
validacao_dados = X[tamanho_do_treino:]
# Separa as marcacoes de validacao
validacao_marcacoes = Y[tamanho_do_treino:]

print("Textos usados na validacao: ")
# print(textoLimpo[tamanho_do_treino:])
print("Validacao Marcacoes: ")
print(validacao_marcacoes)

clf = LogisticRegression()  # MultinomialNB() obtive 62% de acerto#GaussianNB()
t0 = time()
clf.fit(treino_dados, treino_marcacoes)
print("Tempo: ", round(time() - t0, 3), "s")
# resp = clf.predict(validacao_dados)

accuracy = clf.score(validacao_dados, validacao_marcacoes)

print("Indice de acerto do algoritmo: ")
print("%.2f " % round(accuracy * 100) + "%\n")
# salvando  treino com pickle
file_name = 'treino.sav'
pickle._dump(clf, open(file_name, 'wb'))
# salvando treino com joblib
file_name_joblib = 'treino_joblib.sav'
joblib.dump(clf, file_name_joblib)
'''for cat in resp:
    print("CID {:16s}".format(cat))
'''


def predict():
    pr = df.read_csv('csv_to_predict.csv', sep=';', header=0, usecols=[0])
    pred_cols = list(pr.columns.values)[0]
    df['texto'] = vetorizar_texto('buscar resultado',tradutor)
    tvect = TfidfVectorizer(min_df=1, max_df=1)
    X_test = df['texto']  # tvect.transform(test)
    # carregar modelo salvo para predição com pickle
    loaded_model = pickle.load(open(file_name, 'rb'))
    # carregando modelo salvo para predição com joblib
    loaded_model_joblib = joblib.load(file_name_joblib)
    result = loaded_model_joblib.predict(X_test)
    print(result)
predict()
