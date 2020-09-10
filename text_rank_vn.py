import numpy as np
from .utils import get_stop_words_list, get_vocab, get_token_pairs, symmetrize, get_matrix, get_keywords
from underthesea import sent_tokenize, word_tokenize, pos_tag
import re
from langdetect import detect
from vncorenlp import VnCoreNLP
from string import punctuation

class TextRank4KeywordVN():
    """Extract keywords from text"""
    
    def __init__(self, stopwords, ngrams=1, window_size=3, candidate_pos=["N", "Np"], num_keywords=5, use_vncorenlp=True):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight
        self.ngrams = ngrams
        self.window_size = window_size
        self.candidate_pos = candidate_pos
        self.num_keywords = num_keywords
        self.stopwords = stopwords
        self.use_vncorenlp = use_vncorenlp
        if self.use_vncorenlp:
            self.annotator = VnCoreNLP("/home/vudat1710/Downloads/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg,pos", max_heap_size='-Xmx2g')
    
    # tokenizing, filtering stopwords for sentence
    def filtering_sentence(self, sent, stopwords, keyword, lower=False):
        sent = re.sub(r'[^\w\s]','',sent)
        filtered_words = []
        if lower:
            words = word_tokenize(sent, format="text").split(" ")
            words = [word.lower() for word in words]
        else:
            words = word_tokenize(sent, format="text").split(" ")
        for word in words:
            if word not in stopwords and (keyword not in word) and (word not in keyword) and not (word.isnumeric()) and word not in punctuation:
                try:
                    if detect(word) == "vi":
                        filtered_words.append(word)   
                except Exception:
                    continue 
                # filtered_words.append(word)
        return ' '.join(filtered_words)
    
    # pos-tagging text for segmentation step
    def pos_tagging_sentence(self, sent):
        if self.use_vncorenlp:
            sent = sent.replace("_", " ")
            temp = self.annotator.annotate(sent)
            return [(element["form"], element["posTag"]) for sent in temp["sentences"] for element in sent]
        else:
            return pos_tag(sent)

    def sentence_segment(self, doc, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc:
            postag = self.pos_tagging_sentence(sent)
            words = [x[0] for x in postag]
            selected_words = []
            res = []
            for i in range(len(words)):
                # Store words only with cadidate POS tag
                if postag[i][1] in self.candidate_pos or words[i] in ["trình_duyệt"]:
                    if lower is True:
                        selected_words.append(words[i].lower())
                    else:
                        selected_words.append(words[i])
            if self.ngrams == 1:
                sentences.append(selected_words)
            else:
                for i in range(len(selected_words) - self.ngrams + 1):
                    word = ''
                    for j in range(self.ngrams):
                        word += selected_words[i+j]
                        if j != self.ngrams - 1:
                            word += ' '
                    res.append(word)
                sentences.append(res)
        return sentences
        
    def analyze(self, text, keyword, lower=False):
        """Main function to analyze text"""
        doc = sent_tokenize(text)
        doc = [self.filtering_sentence(sent, self.stopwords, keyword, lower) for sent in doc]

        # Filter sentences
        sentences = self.sentence_segment(doc, lower) # list of list of words
        
        # Build vocabulary
        vocab = get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = get_token_pairs(sentences, self.window_size)
        
        # Get normalized matrix
        g = get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight
        return get_keywords(self.node_weight, self.num_keywords)

if __name__=="__main__":
    stopwords = get_stop_words_list('/home/vudat1710/Works/hostname_topic/vietnamese_stopwords_dash.txt')
    text = "Trình duyệt Cốc Cốc, lướt web theo phong cách Việt"
    keyword = "coccoc"
    tr4wvn = TextRank4KeywordVN(stopwords, ngrams=1, window_size=5, num_keywords=3)
    print(tr4wvn.analyze(text, keyword, lower=True))
