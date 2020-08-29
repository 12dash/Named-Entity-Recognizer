import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


class Net(nn.Module):
    def __init__(self,vocab_size,embedding_dim,lstm_hidden_dim,number_of_tags):
        super(Net, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first = True)
        self.fc= nn.Linear(lstm_hidden_dim, number_of_tags)
    
    def forward(self, s):
        s = self.embedding(s)
        s, _ = self.lstm(s)
        s = s.view(-1, s.shape[2])
        s = self.fc(s)
        
        return F.log_softmax(s, dim = 1)
    
    def loss_fn(self,outputs, labels):
        labels = labels.view(-1)
        mask = (labels >= 0).float()
        #num_tokens = int(torch.sum(mask).data[0])
        outputs = outputs[range(outputs.shape[0]), labels]*mask
        
        return -torch.sum(outputs)/17          
    
def fetch_file(path):
    file = open(path,'rb')
    a,b = pickle.load(file)
    file.close()
    return a,b

def process_script(a):
    x = word2idx['PAD']*np.ones((1,104))
    for i in range(len(a)):
        if(a[i] in word2idx.keys()):
            x[0][i] = word2idx[a[i]]
        else:
            x[0][i] = word2idx['UNK']            
    return x

if __name__ == '__main__':
    idx2word,word2idx =fetch_file('Word Level/word_map.pkl')
    idx2tag,tag2idx = fetch_file('Word Level/tag_map.pkl')
    model = torch.load('Word Level/model.pytorch', map_location = 'cpu')
    a = input("Enter the Sentence : ").split()
    s = process_script(a)
    s = torch.LongTensor(s)
    output = model(s)

    pred = []
    for j in output:
        t = [i.item() for i in j]
        m = max(t)
        pred.append(idx2tag[t.index(m)])
    
    for i in range(s.shape[1]):
        if idx2word[int(s[0][i])] == 'PAD':
            break    
        print(a[i],':',(pred[i]))