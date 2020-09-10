import gensim
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.keyedvectors import KeyedVectors
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import ast, heapq

def cosine_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def f(wlist, model, num):
    wlist = [word for word in wlist if word in model.wv.vocab]
    lenwlist=len(wlist)
    if lenwlist == 1:
        return wlist
    elif lenwlist == 0:
        return []
    avrsim=[]
    #compute cosine similarity between each word in wlist with the other words in wlist  
    for i in range(lenwlist):
        word=wlist[i]
        totalsim=0
        wordembed=model[word] 
        for j in range(lenwlist):
            if i!=j:
                word2embed=model[wlist[j]] 
                totalsim+=cosine_sim(wordembed, word2embed)
        avrsim.append(totalsim/ (lenwlist-1))   
    t = heapq.nlargest(num,avrsim)
    return [wlist[avrsim.index(i)] for i in t]

def grouping(topics):
    res = []
    for _l in topics:
        res.extend(_l)
    return res

model_path_en = "./GoogleNews-vectors-negative300.bin"
model_en = KeyedVectors.load_word2vec_format(model_path_en, binary=True)
model_path_vi = "./baomoi.window2.vn.model.bin"
model_vi = KeyedVectors.load_word2vec_format(model_path_vi, binary=True)

df = pd.read_csv("result.csv", converters={"topics_from_en": ast.literal_eval, "topics_from_vi": ast.literal_eval})
vi = df.groupby("keyword").apply(lambda x: grouping(x.topics_from_vi.tolist())).reset_index(level=0)
df["res_en"] = df["topics_from_en"].apply(lambda x: f(x, model_en, 2))
en = df.groupby("keyword").apply(lambda x: grouping(x.res_en.tolist())).reset_index(level=0)
df = pd.merge(en, vi, on="keyword")
df.columns = ["keyword", "en", "vi"]
df["en"] = df["en"].apply(lambda x: list(set(x)))
df["vi"] = df["vi"].apply(lambda x: f(x, model_vi, 3))
df["vi"] = df["vi"].apply(lambda x: list(set(x)))
df.to_csv("res_topics.csv", index=False)

df = pd.read_csv("result_web_title.csv", converters={"topics_from_en": ast.literal_eval, "topics_from_vi": ast.literal_eval})
vi = df.groupby("keyword").apply(lambda x: grouping(x.topics_from_vi.tolist())).reset_index(level=0)
df["res_en"] = df["topics_from_en"].apply(lambda x: f(x, model_en, 2))
en = df.groupby("keyword").apply(lambda x: grouping(x.res_en.tolist())).reset_index(level=0)
df = pd.merge(en, vi, on="keyword")
df.columns = ["keyword", "en", "vi"]
df["en"] = df["en"].apply(lambda x: list(set(x)))
df["vi"] = df["vi"].apply(lambda x: f(x, model_vi, 3))
df["vi"] = df["vi"].apply(lambda x: list(set(x)))
df.to_csv("res_topics_web_title.csv", index=False)