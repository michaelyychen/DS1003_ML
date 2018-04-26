import pandas as pd
import numpy as np
import math
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from preprocessing_data import load_problem
base_dir = "Data/"
filename = "save.pickle"
x_train, y_train, x_test,y_test = load_problem(base_dir+filename)

label2one = {'B':0,'S':1,'X':2}
one2label = {0:'B', 1:'S', 2:'X'}

def normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mu, std, (data-mu)/std

vfunc = np.vectorize(lambda x:label2one[x])

input_labels = ['pitcher','batter','x0','x','y','ax','ay','az','px','pz','sz_top','sz_bot',
             'vx0','vy0','vz0','pfx_x','z0','start_speed','end_speed',
             'break_y','break_angle','break_length','spin_dir','spin_rate',
             'inning','balls','strikes'
             ]
#input_labels = ['pitcher','batter','balls', 'strikes','inning','pitch_count']
#input_labels = ['pitcher','batter', 
       #'on_1b', 'on_2b', 'on_3b', 'pitch_type', 'side', 
       #'inning', 'pitch_count', 'balls', 'strikes','offense_score', 'defense_score', 
#       'ay', 'px', 'ax',  
#       'sz_bot', 'vz0', 'vy0', 'pfx_x',
#       'type_confidence', 'z0', 'tfs', 'pz', 'start_speed', 'az', 'zone',
#       'break_angle', 'spin_dir', 'end_speed', 'vx0', 'sz_top', 'nasty',
#       'pfx_z', 'break_y', 'x', 'spin_rate',
#       'y0', 'break_length', 'y', 'x0'
#       ]
feature_length = len(input_labels)-2
print("Feature length:{}".format(feature_length))
train_years = [5,6,7]
dev_years = [7]


train_x = {}
train_y = {}
ctr = 0
for y in train_years:
    filename= base_dir+"MLB_201{0}/MLB_PitchFX_201{0}_RegularSeason.csv".format(str(y))
    print("Loading {}".format(filename))
    f = pd.read_csv(filename)
    
    tmp_x = f[input_labels]
    tmp_y = f['umpcall']

    tmp_x = tmp_x.as_matrix()
    tmp_y = tmp_y.as_matrix()
    tmp_y = vfunc(tmp_y)

    if ctr==0:
        ctr=1
        train_x = tmp_x
        train_y = tmp_y
    else:
        print(train_x.shape)
        print(tmp_x.shape)
        train_x = np.concatenate((train_x, tmp_x), axis=0)
        train_y = np.concatenate((train_y, tmp_y), axis=0)

filename = base_dir+"MLB_2017/MLB_PitchFX_2017_PostSeason.csv"
print("Loading dev file {}".format(filename))
f2 = pd.read_csv(filename)
dev_x = f2[input_labels]
dev_y = f2['umpcall']

dev_x = dev_x.as_matrix()
dev_y = dev_y.as_matrix()
dev_y = vfunc(dev_y)

from sklearn.utils import shuffle
train_x, train_y = shuffle(train_x, train_y)

class Lang:
    def __init__(self, name):
        """Init Lang with a name."""
        self.name = name
        self.word2index = {"<UNK>": 0}
        self.word2count = {}
        self.index2word = {0: "<UNK>"}
        self.n_words = 1  # Count SOS and EOS

    def addword(self, word):
        """Add a word to the dict."""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

pitch_lang = Lang('pitcher')
batter_lang = Lang('batter')
pits = train_x[:,0]
bats = train_x[:,1]
for i in range(len(train_x)):
    pitch_lang.addword(pits[i])
    batter_lang.addword(bats[i])

def map2idx(train_x):
    pi = train_x[:,0]
    vfnc = np.vectorize(lambda x: pitch_lang.word2index[x] if x in pitch_lang.word2index else 0)
    pi = vfnc(pi).reshape(-1,1)
    vfnc = np.vectorize(lambda x:batter_lang.word2index[x] if x in batter_lang.word2index else 0)
    ba = vfnc(train_x[:,1]).reshape(-1,1)
    
    return np.concatenate((pi,ba,train_x[:,2:]), axis=1)

class PlayerEmbedding(nn.Module):
    def __init__(self, pitcher_size, batter_size, embedding_dim):
        super(PlayerEmbedding, self).__init__()
        self.embedding1 = nn.Embedding(pitcher_size, embedding_dim)
        self.embedding2 = nn.Embedding(batter_size, embedding_dim)
        self.out_dim = embedding_dim*16
        self.conv1 = nn.Conv1d(2,32,3, padding=1)

        self.conv2 = nn.Conv1d(32,32,3, padding=1)
        self.conv3 = nn.Conv1d(32,32,3, padding=1)
        self.maxpol = nn.MaxPool1d(2)
        self.embedding_dim = embedding_dim

    def forward(self, p, b):
        emb_p = self.embedding1(p)
        emb_b = self.embedding2(b)
        emb_p = emb_p.unsqueeze(1)
        emb_b = emb_b.unsqueeze(1)

        emb_all = torch.cat([emb_p, emb_b], dim=1)
        bypass = F.relu(self.conv1(emb_all))
        
        emb_all = F.relu(self.conv2(bypass))
        emb_all = F.relu(self.conv3(emb_all))
        emb_all += bypass
        emb_all = self.maxpol(emb_all)
        emb_all = emb_all.view(emb_all.shape[0],-1)
        
        return emb_all

    def init_weights(self):
        initrange = 0.5
        em_layer = [self.embedding1, self.embedding2]

        for layer in em_layer:
            #layer.weight.data.normal_(0, initrange)
            layer.weight.data.uniform_(-initrange, initrange)

class ResNet(nn.Module):
    def __init__(self, in_dim):
        super(ResNet, self).__init__()

        self.lin1 = nn.Linear(in_dim, in_dim)
        self.lin2 = nn.Linear(in_dim, in_dim)
    def forward(self, x):
        output = F.relu(self.lin1(x))
        return self.lin2(output) + x
class Fcc(nn.Module):
    def __init__(self, embedding_layer):
        super(Fcc, self).__init__()

        self.embedding = embedding_layer
        #hid_dim = [100,100,100,100,100]
        p = 0.0
        res_dim=1024
        expand_fea_dim = 64

        self.bn = torch.nn.BatchNorm1d(feature_length)
        #self.drop1 = nn.Dropout(p)
        #self.lin1 = nn.Linear(2*self.embedding.embedding_dim + 22, hid_dim[0])

        self.fea1 = nn.Linear(feature_length,64)
        #self.feadrop1 = nn.Dropout(p)
        self.feares2 = ResNet(64)
        #self.feadrop2 = nn.Dropout(p)
        self.fea3 = nn.Linear(64,expand_fea_dim)

        self.lin1 = nn.Linear(self.embedding.out_dim + expand_fea_dim, res_dim)
        #self.lin1 = nn.Linear(2*self.embedding.embedding_dim, hid_dim[0])
        #self.drop2 = nn.Dropout(p)
        self.res2 = ResNet(res_dim)
        #self.drop3 = nn.Dropout(p)
        self.res3 = ResNet(res_dim)
        #self.drop4 = nn.Dropout(p)

        self.lin3 = nn.Linear(res_dim, 256)

        self.res4 = ResNet(256)
        #self.drop5 = nn.Dropout(p)
        self.res5 = ResNet(256)
        #self.drop6 = nn.Dropout(p)
        #self.lin6 = nn.Linear(256, 64)
        #self.lin7 = nn.Linear(64,64)
        self.lin8 = nn.Linear(256,3)
    
    def forward(self, players, features):
        embedded = F.relu(self.embedding(players[:,0], players[:,1]))
        fea = self.bn(features)
        fea = F.relu(self.fea1(fea))
        #fea = self.feadrop1(fea)
        fea = F.relu(self.feares2(fea))
        #fea = self.feadrop2(fea)
        fea = F.relu(self.fea3(fea))
        output = torch.cat([embedded, fea], dim=1)
        #output = embedded

        #output = self.drop1(output)
        output = self.lin1(output)
        output = F.relu(output)
        
        #output = self.drop2(output)
        output = self.res2(output)
        output = F.relu(output)
        
        #output = self.drop3(output)
        output = self.res3(output)
        output = F.relu(output)
        
        output = F.relu(self.lin3(output))

        #output = self.drop4(output)
        output = self.res4(output)
        output = F.relu(output)

        #output = self.drop5(output)
        output = self.res5(output)
        output = F.relu(output)

        #output = self.lin6(output)
        #output = F.relu(output)
        #output = self.lin7(output)
        #output = F.relu(output)

        output = self.lin8(output)
        output = F.log_softmax(output, dim=1)

        return output

def data_gen(dx, df, dy, batch_size=100):
    idx = 0
    while True:
        if idx*batch_size >= dx.shape[0]:
            return
        elif (idx+1)*batch_size > dx.shape[0]:
            yield dx[idx*batch_size:,:], df[idx*batch_size:,:], dy[idx*batch_size:]
        else:
            yield dx[idx*batch_size:(idx+1)*batch_size,:], df[idx*batch_size:(idx+1)*batch_size], dy[idx*batch_size:(idx+1)*batch_size]
        idx += 1



vdx = Variable(torch.from_numpy(map2idx(dev_x[:,:2]).astype(np.long))).cuda()
vdf = Variable(torch.from_numpy(dev_x[:,2:].astype(np.float32))).cuda()
vdy = Variable(torch.from_numpy(dev_y)).cuda()

def train(train_x, train_y, dev_x, dev_f, dev_y, model, optimizer, criterion, batch_size=512, max_epoch = 512, validation_interv=1000):
    print(train_x.shape)
    for ep in range(max_epoch):
        train_x, train_y = shuffle(train_x, train_y)
        vtx = Variable(torch.from_numpy(map2idx(train_x[:,:2]).astype(np.long)), requires_grad=False).cuda()
        vtf = Variable(torch.from_numpy(train_x[:,2:].astype(np.float32)), requires_grad=False).cuda()
        vty = Variable(torch.from_numpy(train_y), requires_grad=False).cuda()
        print("Epoch {}".format(ep))
        train_iter = data_gen(vtx, vtf, vty, batch_size=batch_size)
        ctr = 1
        avg_loss = 0
        for bx,bf,by in train_iter:
            optimizer.zero_grad()
            model.train()
            y_pred = model(bx,bf)
            loss = criterion(y_pred, by)
            loss.backward()
            avg_loss += loss.data[0]
            optimizer.step()
            if ctr%validation_interv==0:
                model.eval()
                ll = 0
                ctrr = 0
                for dx,df,dy in data_gen(dev_x, dev_f, dev_y, batch_size=batch_size):
                    dy_pred = model(dx, df)
                    tmp = criterion(dy_pred, dy).data[0]
                    ll += tmp * dx.shape[0]
                    ctrr += dx.shape[0]
                print("loss:{} dev_loss:{}".format(avg_loss/ctr, ll/ctrr))
            ctr+=1
        del vtx, vtf, vty, train_iter

bt_size = 2048
emb_layer = PlayerEmbedding(pitch_lang.n_words,batter_lang.n_words, 200)
model = Fcc(emb_layer).cuda()
print(model)
#opt = torch.optim.SGD(model.parameters(), lr=1e-3,momentum=0.9)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
crit = nn.NLLLoss()
train(train_x, train_y, vdx, vdf, vdy, model, opt, crit, batch_size=bt_size, max_epoch=30, validation_interv=100)

model.eval()
ll=0
ctrr=0
for dx,df,dy in data_gen(vdx, vdf, vdy, batch_size=bt_size):
    dy_pred = model(dx, df)
    pred = dy_pred.exp().cpu().data.numpy()
    y_pred = np.argmax(pred, axis=1)
    tmp = np.sum( y_pred == dy.cpu().data.numpy())
    ll += tmp
    ctrr += dx.shape[0]

print(ll/ctrr)