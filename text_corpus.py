from nltk import word_tokenize, sent_tokenize
from utils import cosine_similarity
from gensim.models import Word2Vec 

class TextCorpusSearcher:
    def __init__(self, filename, x, label):
        self.label = label
        self.x = tuple(word.lower() for word in x)
        text = self.get_text(filename)
        sentences = []
        for sent in sent_tokenize(text.lower()):
            s = []
            for word in word_tokenize(sent):
                s.append(word)
            sentences.append(s)

        self.model = Word2Vec(sentences, vector_size=12, window=3, min_count=1, sg=0)

        for w in self.x:
            if w not in self.model.wv:
                print("[WARN]", w, "missing in Word2Vec training data")

    def get_text(self, filename):
        with open(filename) as f:
            return f.read() 
    
    def get_embed(self, word):
        return self.model.wv[word]

    def get_score(self, word):
        max_score = 0
        global curr_model
        word = word.lower()
        curr_model = self.model
        if word not in self.model.wv:
            return 0, self.label
        for w in self.x:
            score = cosine_similarity(self.model.wv[word], self.model.wv[w])
            max_score = max(max_score, score)
        return max_score, self.label