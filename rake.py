from rake_nltk import Metric, Rake
from underthesea import word_tokenize, sent_tokenize, pos_tag
from .utils import get_stop_words_list
from string import punctuation
import re

def filtering_sentence(sent, stopwords, keyword, lower=False):
    sent = re.sub(r'[^\w\s]','',sent)
    filtered_words = []
    if lower:
        words = word_tokenize(sent, format="text").split(" ")
        words = [word.lower() for word in words]
    else:
        words = word_tokenize(sent, format="text").split(" ")
    for word in words:
        if word not in stopwords and word != keyword and not word.isnumeric():
            filtered_words.append(word)    
    return ' '.join(filtered_words)

def sentence_segment(doc, lower, candidate_pos):
    """Store those words only in cadidate_pos"""
    sentences = []
    for sent in doc:
        postag = pos_tag(sent)
        words = [x[0] for x in postag]
        selected_words = []
        res = []
        for i in range(len(words)):
            # Store words only with cadidate POS tag
            if postag[i][1] in candidate_pos:
                if lower is True:
                    selected_words.append(words[i].lower())
                else:
                    selected_words.append(words[i])
        sentences.append(' '.join(selected_words))
        
    return '. '.join(sentences)

stoppath = '/home/vudat1710/Works/hostname_topic/vietnamese_stopwords_dash.txt'
text = "Cập nhật tin tức sự kiện, báo mới nhất ở VN, các TIN NHANH độc quyền phản ánh đầy đủ chuyển động kinh tế & xã hội Việt Nam, thế giới trong 24h qua."
keyword = "soha"
stopwords = get_stop_words_list(stoppath)
print(len(stopwords))
r = Rake(stopwords=stopwords, ranking_metric=Metric.WORD_DEGREE)
candidate_pos = ["N", "FW", "B", "Nb", "Np"]
text = sent_tokenize(text)
text = [filtering_sentence(sent, stopwords, lower=True, keyword="") for sent in text]
text = sentence_segment(text, True, candidate_pos)
print(text)
print(r.extract_keywords_from_text(text))

