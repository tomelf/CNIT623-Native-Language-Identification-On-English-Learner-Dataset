import re

class FunctionWords(object):
    def __init__(self, filename=None):
        self.features = None
        if filename != None:
            self.set_features_path(filename)
        
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
        sent_in_words = sent_in_sent.split(" ")
        ret_vec = []
        for fw in self.features:
            if len(fw) == 0:
                continue
            if len(fw.split(" ")) > 1:
                ret_vec.append(len(re.findall(fw, sent_in_sent)))
            else:
                ret_vec.append(sent_in_words.count(fw))
        return ret_vec
    
    def get_features(self):
        return self.features