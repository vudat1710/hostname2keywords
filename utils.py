from collections import OrderedDict
import numpy as np

def get_stop_words_list(filename):
    stopwords = []
    with open(filename, "r") as f:
        for line in f.readlines():
            stopwords.append(line.strip())
    f.close()
    return stopwords

def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def get_vocab(sentences):
    """Get all tokens"""
    vocab = OrderedDict()
    i = 0
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = i
                i += 1
    return vocab

def get_token_pairs(sentences, window_size):
    """Build token_pairs from windows in sentences"""
    token_pairs = list()
    for sentence in sentences:
        for i, word in enumerate(sentence):
            for j in range(i+1, i+window_size):
                if j >= len(sentence):
                    break
                pair = (word, sentence[j])
                if pair not in token_pairs:
                    token_pairs.append(pair)
    return token_pairs

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

def get_matrix(vocab, token_pairs):
    """Get normalized matrix"""
    # Build matrix
    vocab_size = len(vocab)
    g = np.zeros((vocab_size, vocab_size), dtype='float')
    for word1, word2 in token_pairs:
        i, j = vocab[word1], vocab[word2]
        g[i][j] = 1
        
    # Get Symmeric matrix
    g = symmetrize(g)
    
    # Normalize matrix by column
    norm = np.sum(g, axis=0)
    g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
    
    return g_norm


def get_keywords(node_weight, num_keywords):
    """Print top number keywords"""
    return [k[0] for k in sorted(node_weight.items(), key=lambda t: t[1], reverse=True)[0:num_keywords]]


def merge_results():
    df1 = pd.read_csv("res_topics.csv", converters={"en": ast.literal_eval, "vi": ast.literal_eval})
    df2 = pd.read_csv("res_topics_web_title.csv", converters={"en": ast.literal_eval, "vi": ast.literal_eval})
    df2 = df2.rename(columns={"en": "en_title"})
    df2 = df2.rename(columns={"vi": "vi_title"})
    df = pd.concat([df1, df2["en_title"], df2["vi_title"]], axis=1)
    df["en"] = df["en"] + df["en_title"]
    df["vi"] = df["vi"] + df["vi_title"]
    df["en"] = df["en"].apply(lambda x: list(set(x)))
    df["vi"] = df["vi"].apply(lambda x: list(set(x)))
    df = df.drop(["en_title", "vi_title"], axis=1)
    df.to_csv("final_keywords.csv", index=False)