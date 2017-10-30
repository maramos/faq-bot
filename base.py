import csv
import matplotlib.pyplot as mp
import nltk as nltk
import sklearn
import scipy as sp
import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

json = open("tmp", "r")

class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        global stopwords
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc:(nltk.stem.RSLPStemmer().stem(w) for w in analyzer(doc) if w not in stopwords)

def convert(csv):
    quest = []
    answer = []
    for i in csv:
        quest.append(i.split(";")[0])
        answer.append(i.split(";")[1])
    dictcsv = {i:j for i, j in zip(quest, answer)}
    return dictcsv


def get_quest(hash):
    get_quest_saida = []
    for i,j in hash.items():
        get_quest_saida.append(stemmer.stem(i))
    return get_quest_saida


def distance(v1, v2):
    delta = (v1/sp.linalg.norm(v1.toarray()))-(v2/sp.linalg.norm(v2.toarray()))
    return sp.linalg.norm(delta.toarray())


def nearest_one(quests, new_quest):
    best_doc = None
    best_dist = sys.maxint
    best_i = None

    for i in range(0, n_samples):
        quest = quests
        if quest==new_quest:
            continue

        dist = distance(X_train.getrow(i), vec.transform([new_quest]))
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i


def get_ans(new_quest):
    nearest_question = nearest_one(FAQ.keys(),new_quest)
    return FAQ.items()[nearest_question][1]


FAQ = convert(json)
stemmer = nltk.stem.RSLPStemmer()
stopwords = nltk.corpus.stopwords.words('portuguese')

#vec = sklearn.feature_extraction.text.CountVectorizer(min_df=1)
#vec = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1)
vec = TfidfVectorizer(min_df=1)

X_train = vec.fit_transform(get_quest(FAQ))
n_samples, n_features = X_train.shape

def nt(text):
    new_text = text
    new_text_vec = vec.transform([new_text])
    nearest_question = nearest_one(FAQ.keys(),new_text)
    FAQ.items()[nearest_question][1]
    
    
######
#new_text = "gostando"
#new_text_vec = vec.transform([new_text])
#nearest_question = nearest_one(FAQ.keys(),new_text)
#FAQ.items()[nearest_question][1]
######
