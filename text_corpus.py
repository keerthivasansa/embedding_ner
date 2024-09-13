from numpy import mean
import spacy

from utils import cosine_similarity

class TextCorpusSearcher:
    def __init__(self, filename, x, label):
        
        self.label = label
        nlp = spacy.load('en_core_web_sm')
        doc = ' '.join(x).lower()
        # self.stemmer = SnowballStemmer(language='english')
        self.x = [word.text for word in nlp(doc) if not (word.is_stop or word.is_space or word.is_punct)]
        print(self.x)
        text = self.get_text(filename)
        sentences = []
        
        for sent in sent_tokenize(text.lower()):
            s = []
            for word in nlp(sent):
                if word.is_stop or word.is_punct or word.is_space:
                    continue
                s.append(word.text)
            sentences.append(s)

        self.model = Word2Vec(sentences, vector_size=50, window=3, min_count=4, sg=1)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=100)

        for w in self.x:
            if w not in self.model.wv:
                print("[WARN]", w, "missing in Word2Vec training data")

    def get_text(self, filename):
        with open(filename) as f:
            return f.read() 
    
    def get_embed(self, word):
        return self.model.wv[word.lower()]
    
    def has_embed(self, word):
        return word in self.model.wv

    def get_score(self, word):
        global curr_model
        word = word.lower()
        tag = pos_tag([word])[0][1]
        print(tag, word)
        empty = np.zeros(len(self.x))

        if not tag.startswith('NN') or word not in self.model.wv:
            return empty
        
        curr_model = self.model
        scores = []
        for w in self.x:
            if w not in self.model.wv:
                continue
            score = cosine_similarity(self.model.wv[word], self.model.wv[w])
            scores.append(score)
        print(scores)
        return scores
    