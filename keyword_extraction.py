import json
import pandas as pd
from underthesea import word_tokenize
from langdetect import detect
from .text_rank_en import TextRank4KeywordEN
from .text_rank_vn import TextRank4KeywordVN
import spacy, ast
from .utils import get_stop_words_list
from tldextract import extract
from nltk.stem import PorterStemmer
import numpy as np

def detect_lang(text):
    try:
        return detect(text)
    except Exception:
        return ""

def read_gg_search_data(filename):
    df = pd.DataFrame(json.load(open(filename, "r")))
    df["count"] = df.groupby(by=["keyword"])["keyword"].transform("count")
    df = df.reset_index().sort_values(by=["count"], ascending=False)
    df = df[~((df["count"] == 1) & ((df["url"] == "https://vi.wikipedia.org") | (df["url"] == "https://en.wikipedia.org")))]
    df["desc_lang"] = df["description"].apply(lambda x: detect_lang(x))
    df["title_lang"] = df["title"].apply(lambda x: detect(x))
    return df

def get_topics(text, lang, tr_en, tr_vn, keyword, desc):
    if lang == "en":
        # return ", ".join(tr_en.analyze(text, keyword, lower=True))
        try: 
            if np.isnan(float(text)):
                return tr_en.analyze(desc, keyword, lower=True)
        except Exception:
            return tr_en.analyze(text, keyword, lower=True)
    elif lang == "vi":
        # return ", ".join(tr_vn.analyze(text, keyword, lower=True))
        try:
            if np.isnan(float(text)):
                return tr_vn.analyze(desc, keyword, lower=True)
        except Exception:
            return tr_vn.analyze(text, keyword, lower=True)
    else:
        # return "can't extract topics from this language"
        return []

def get_topics_web_title(web_title, lang, tr_en, tr_vi, keyword):
    if lang == "en":
        return tr_en.analyze(web_title, keyword, lower=True)
    elif lang == "vi":
        return tr_vn.analyze(web_title, keyword, lower=True)
    else:
        return []

def filter_topics_list(topics_lang_list, porter):
    res = {"en": [], "vi": []}
    topics_list = [x["topics_list"] for x in topics_lang_list]
    lang_list = [x["lang"] for x in topics_lang_list]
    en_topics = []
    for i in range(len(topics_list)):
        if lang_list[i] == "en":
            en_topics.extend(topics_list[i])
        elif lang_list[i] == "vi":
            res["vi"].extend(topics_list[i])
    en_topics = list(set(en_topics))
    _d = {}
    for topic in en_topics:
        stem_topic = porter.stem(topic)
        if stem_topic not in _d:
            _d[stem_topic] = []
        _d[stem_topic].append(topic)
    res["en"] = [_d[k][-1] for k in _d]
    res["vi"] = list(set(res["vi"]))

    return res

def grouping_data(keyword, url_list, topics_list, desc_lang_list):
    _d = {}
    res = []
    porter = PorterStemmer()
    for i in range(len(url_list)):
        _, td, ts = extract(url_list[i])
        new_url = ".".join([td, ts])
        if new_url not in _d:
            _d[new_url] = []
        _d[new_url].append({"topics_list": topics_list[i], "lang": desc_lang_list[i]})
    for k in _d:
        _d[k] = filter_topics_list(_d[k], porter)
        res.append({"keyword": keyword, "url": k, "topics_from_en": _d[k]["en"], "topics_from_vi": _d[k]["vi"]})
    
    return res

if __name__=="__main__":
    # df = read_gg_search_data("/home/vudat1710/Works/google_scraper/google_scraper/data/gg_search.json")
    # df.to_csv("gg_search_lang.csv", index=False)


    stopwords = get_stop_words_list('/home/vudat1710/Works/hostname_topic/vietnamese_stopwords_dash.txt')
    nlp = spacy.load('en_core_web_sm')
    tr_en = TextRank4KeywordEN(nlp, ngrams=1, window_size=5, num_keywords=5)
    tr_vn = TextRank4KeywordVN(stopwords, ngrams=1, window_size=4, num_keywords=7, use_vncorenlp=True)
    df = pd.read_csv("/home/vudat1710/Works/hostname_topic/data/merge_crawled_data.csv")
    df["text_topics"] = df.apply(lambda x: get_topics(x.merged_text, x.desc_lang, tr_en, tr_vn, x.keyword, x.description), axis=1)
    df["web_title_topics"] = df.apply(lambda x: get_topics_web_title(x.web_title, x.web_title_lang, tr_en, tr_vn, x.keyword), axis=1)
    df.to_csv("gg_search_topics.csv", index=False)
    
    
    df = pd.read_csv("/home/vudat1710/Works/gg_search_topics.csv", converters={'text_topics': ast.literal_eval, "web_title_topics": ast.literal_eval})
    keywords = df["keyword"].unique().tolist()
    
    result = []
    
    for keyword in keywords:
        temp = df[df["keyword"] == keyword]
        url_list = temp["url"].tolist()
        topics_list = temp["text_topics"].tolist()
        desc_lang_list = temp["desc_lang"].tolist()
        result.extend(grouping_data(keyword, url_list, topics_list, desc_lang_list))
    
    pd.DataFrame(result).to_csv("result.csv", index=False)

    result = []
    
    for keyword in keywords:
        temp = df[df["keyword"] == keyword]
        url_list = temp["url"].tolist()
        topics_list = temp["web_title_topics"].tolist()
        desc_lang_list = temp["web_title_lang"].tolist()
        result.extend(grouping_data(keyword, url_list, topics_list, desc_lang_list))
    
    pd.DataFrame(result).to_csv("result_web_title.csv", index=False)
