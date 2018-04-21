import re
from itertools import product
from string import ascii_lowercase

class CharNGrams(object):
    def __init__(self, n=3):
        self.features = [''.join(i) for i in product(ascii_lowercase, repeat = n)]
        
    def set_features_path(self, filename):
        features_set = set()
        with open(filename, "r") as f:
            for line in f:
                word = line.strip()
                if len(word)>0:
                    features_set.add(word)
        self.features = list(features_set)
        
    def sent2vec(self, sent):
        sent_in_sent = re.sub("[^\w ]", "", sent.lower())
        ret_vec = []
        for fw in self.features:
            ret_vec.append(len(re.findall(fw, sent_in_sent)))
        return ret_vec
    
    def get_features(self):
        return self.features