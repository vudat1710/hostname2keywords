import numpy as np
import spacy, re
from .utils import is_english, get_vocab, get_token_pairs, symmetrize, get_matrix, get_keywords
from spacy.lang.en.stop_words import STOP_WORDS

class TextRank4KeywordEN():
    """Extract keywords from text"""
    
    def __init__(self, nlp_spacy, ngrams=1, window_size=4, candidate_pos=["NOUN", "PROPN"], num_keywords=3):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 20 # iteration steps
        self.node_weight = None # save keywords and its weight
        self.ngrams = ngrams
        self.window_size = window_size
        self.candidate_pos = candidate_pos
        self.num_keywords = num_keywords
        self.nlp = nlp_spacy
    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        stopwords.extend(["right", "left", "tv", "console", "pc", "users", "web", "js", "vietnam", "url", "-", "password", "forgot", "login", "|", "policy", "privacy", "terms", "c", "d", "hoi","email", "mail", "website", "user", "sign", "tomorrow", "day", "today", "yesterday", "nam", "javascript"])
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = self.nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, lower, keyword):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            res = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in self.candidate_pos and token.is_stop is False and token.text.lower() != keyword and is_english(token.text) and not token.text.isnumeric() and "(" not in token.text and len(token) > 1:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
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
                
    def analyze(self, text, keyword, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        # Pare text by spaCy
        if lower:
            text = text.lower()
        doc = self.nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, lower, keyword) # list of list of words
        
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
    nlp = spacy.load('en_core_web_sm')
    text = "Netflix - Watch TV Shows Online, Watch Movies Online. Watch Netflix movies & TV shows online or stream right to your smart TV, game console, PC, Mac, mobile, tablet and more."
    keyword = 'netflix'
    tr4w = TextRank4KeywordEN(nlp, ngrams=1, window_size=5, num_keywords=3)
    print(tr4w.analyze(text, keyword, lower=True))