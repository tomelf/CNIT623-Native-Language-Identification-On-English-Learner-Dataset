import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from gensim.models import KeyedVectors
from sklearn import metrics

class LSTMNLI(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_dim, bidirectional=False):
        super(LSTMNLI, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_dim = label_dim
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        self.hidden2tag = nn.Linear(hidden_dim*(2 if bidirectional else 1), label_dim)
        self.hidden = self.init_hidden()

    def google_w2v_embedding(self, word):
        if not hasattr(self, 'w2v'):
            print("Loading word2vec model: GoogleNews-vectors-negative300.bin")
            self.w2v = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        return self.w2v[word].tolist() if word in self.w2v else self.w2v["_"].tolist()
    
    def init_hidden(self):
        return (autograd.Variable(torch.zeros((2 if self.bidirectional else 1), 1, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros((2 if self.bidirectional else 1), 1, self.hidden_dim)).cuda())

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(
            sentence.view(len(sentence), 1, -1), self.hidden)
        label_space = self.hidden2tag(lstm_out[-1])
        pred = F.softmax(label_space, dim=1)
        return pred

    def prepare_sequence(self, seq):
        idxs = [self.google_w2v_embedding(w) for w in seq]
        tensor = torch.FloatTensor(idxs).cuda()
        return autograd.Variable(tensor).cuda()
    
    def label2onehot(self, label):
        one_hot = [1 if (i+1)==label else 0 for i in range(self.label_dim)]
        tensor = torch.FloatTensor([one_hot]).cuda()
        return autograd.Variable(tensor).cuda()
    
    def set_train_data(self, X_train):
        self.X_train = X_train

    def set_dev_data(self, X_dev):
        self.X_dev = X_dev

    def train(self, epoch=300, lr=0.5):
        loss_function = nn.MSELoss().cuda()
        # loss_function = nn.NLLLoss().cuda()
        # loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        self.train_loss = []
        self.dev_loss = []
        self.train_accuracy = []
        self.dev_accuracy = []

        print("Start model training")
        for e in range(epoch):
            running_loss = 0.0
            for i in range(len(self.X_train)):
                self.zero_grad()
                self.hidden = self.init_hidden()
                
                actual = self.label2onehot(self.X_train[i][1])
                pred = self.test(self.X_train[i][0])
                
                loss = loss_function(pred, actual)
                running_loss += loss.data[0]
                
                loss.backward()
                optimizer.step()
            print("running_loss: {}".format(running_loss))

            if (e+1)%1 == 0:
                print("Finish training epochs {}".format(e+1))

            if hasattr(self, 'X_dev'):
                model = self

                print("Evalute train loss and accuracy")
                actuals = []
                preds = []
                running_loss = 0.0
                for i in range(len(self.X_train)):
                    actual = self.label2onehot(self.X_train[i][1])
                    pred = model.test(self.X_train[i][0])
                    
                    loss = loss_function(pred, actual)
                    running_loss += loss.data[0]
                    
                    _, pred = torch.max(pred, 1)
                    preds.append(int(pred.data.cpu()))
                    actuals.append(actual)
                    
                self.train_loss.append(running_loss/len(self.X_train))
                self.train_accuracy.append(metrics.accuracy_score(actuals, preds))

                print("Evalute dev loss and accuracy")
                actuals = []
                preds = []
                running_loss = 0.0
                for i in range(len(self.X_dev)):
                    actual = self.label2onehot(self.X_dev[i][1])
                    pred = model.test(self.X_dev[i][0])
                    
                    loss = loss_function(pred, actual)
                    running_loss += loss.data[0]
                    
                    _, pred = torch.max(pred, 1)
                    preds.append(int(pred.data.cpu()))
                    actuals.append(actual)
                    
                self.dev_loss.append(running_loss/len(self.X_dev))
                self.dev_accuracy.append(metrics.accuracy_score(actuals, preds))

                pd.DataFrame({
                    "train_loss": self.train_loss,
                    "train_accuracy": self.train_accuracy,
                    "dev_loss": self.dev_loss,
                    "dev_accuracy": self.dev_accuracy,
                }).to_csv("training_stats.csv")

    def test(self, sent_vec):
        sent = self.prepare_sequence(sent_vec)
        return self(sent)

    def get_training_loss(self):
        return self.training_loss

    
class LSTMNLIPOS(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_dim, pos_to_ix, bidirectional=False):
        super(LSTMNLIPOS, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_dim = label_dim
        self.bidirectional = bidirectional
        self.pos_to_ix = pos_to_ix
        self.pos_to_ix_list = [t[0] for t in sorted(pos_to_ix.items(), key=lambda k:k[1])]
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        self.hidden2tag = nn.Linear(hidden_dim*(2 if bidirectional else 1), label_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros((2 if self.bidirectional else 1), 1, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros((2 if self.bidirectional else 1), 1, self.hidden_dim)).cuda())

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(
            sentence.view(len(sentence), 1, -1), self.hidden)
        label_space = self.hidden2tag(lstm_out[-1])
        pred = F.softmax(label_space, dim=1)
        return pred

    def prepare_sequence(self, seq):
        def one_hot(w):
            return [1 if w==pos else 0 for pos in self.pos_to_ix_list]
        idxs = [one_hot(w) for w in seq if w in self.pos_to_ix]
        tensor = torch.FloatTensor(idxs).cuda()
        return autograd.Variable(tensor).cuda()
    
    def label2onehot(self, label):
        one_hot = [1 if (i+1)==label else 0 for i in range(self.label_dim)]
        tensor = torch.FloatTensor([one_hot]).cuda()
        return autograd.Variable(tensor).cuda()
    
    def set_train_data(self, X_train):
        self.X_train = X_train

    def set_dev_data(self, X_dev):
        self.X_dev = X_dev

    def train(self, epoch=300, lr=0.5):
        loss_function = nn.MSELoss().cuda()
        # loss_function = nn.NLLLoss().cuda()
        # loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        self.train_loss = []
        self.dev_loss = []
        self.train_accuracy = []
        self.dev_accuracy = []

        print("Start model training")
        for e in range(epoch):
            running_loss = 0.0
            for i in range(len(self.X_train)):
                self.zero_grad()
                self.hidden = self.init_hidden()
                
                actual = self.label2onehot(self.X_train[i][1])
                pred = self.test(self.X_train[i][0])
                
                loss = loss_function(pred, actual)
                running_loss += loss.data[0]
                
                loss.backward()
                optimizer.step()
            print("running_loss: {}".format(running_loss))

            if (e+1)%1 == 0:
                print("Finish training epochs {}".format(e+1))

            if hasattr(self, 'X_dev'):
                model = self

                print("Evalute train loss and accuracy")
                actuals = []
                preds = []
                running_loss = 0.0
                for i in range(len(self.X_train)):
                    actual = self.label2onehot(self.X_train[i][1])
                    pred = model.test(self.X_train[i][0])
                    
                    loss = loss_function(pred, actual)
                    running_loss += loss.data[0]
                    
                    _, pred = torch.max(pred, 1)
                    preds.append(int(pred.data.cpu()))
                    actuals.append(actual)
                    
                self.train_loss.append(running_loss/len(self.X_train))
                self.train_accuracy.append(metrics.accuracy_score(actuals, preds))

                print("Evalute dev loss and accuracy")
                actuals = []
                preds = []
                running_loss = 0.0
                for i in range(len(self.X_dev)):
                    actual = self.label2onehot(self.X_dev[i][1])
                    pred = model.test(self.X_dev[i][0])
                    
                    loss = loss_function(pred, actual)
                    running_loss += loss.data[0]
                    
                    _, pred = torch.max(pred, 1)
                    preds.append(int(pred.data.cpu()))
                    actuals.append(actual)
                    
                self.dev_loss.append(running_loss/len(self.X_dev))
                self.dev_accuracy.append(metrics.accuracy_score(actuals, preds))

                pd.DataFrame({
                    "train_loss": self.train_loss,
                    "train_accuracy": self.train_accuracy,
                    "dev_loss": self.dev_loss,
                    "dev_accuracy": self.dev_accuracy,
                }).to_csv("training_stats.csv")

    def test(self, sent_vec):
        sent = self.prepare_sequence(sent_vec)
        return self(sent)

    def get_training_loss(self):
        return self.training_loss