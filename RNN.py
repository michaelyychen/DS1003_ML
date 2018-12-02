import pandas as pd
import numpy as np
import math
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import argparse

base_dir = "Data/"

label2one = {'B':0,'S':1,'X':2, '<PAD>':3}
one2label = {0:'B', 1:'S', 2:'X', 3:'<PAD>'}

def normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mu, std, (data-mu)/std

vfunc = np.vectorize(lambda x:label2one[x])

#input_labels = ['pitcher','batter', 'pitch_type','x0','x','y','ax','ay','az','px','pz','sz_top','sz_bot',
#             'vx0','vy0','vz0','pfx_x','z0','start_speed','end_speed',
#             'break_y','break_angle','break_length','spin_dir','spin_rate',
#             'inning','balls','strikes'
#             ]
#input_labels = ['pitcher','batter', 'pitch_type','balls', 'strikes','inning','pitch_count']
input_labels = ['date','stadium', 'inning', 'side',
                'pitcher','batter', 
               'on_1b', 'on_2b', 'on_3b', 
               'pitch_count', 'balls', 'strikes',
#       'ay', 'px', 'ax',  
#       'sz_bot', 'vz0', 'vy0', 'pfx_x',
#       'type_confidence', 'z0', 'tfs', 'pz', 'start_speed', 'az', 'zone',
#       'break_angle', 'spin_dir', 'end_speed', 'vx0', 'sz_top', 'nasty',
#       'pfx_z', 'break_y', 'x', 'spin_rate',
#       'y0', 'break_length', 'y', 'x0'
      ]
feature_length = len(input_labels)-3
print("Feature length:{}".format(feature_length))
train_years = [4,5,6]
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
    
    filename= base_dir+"MLB_201{0}/MLB_PitchFX_201{0}_PostSeason.csv".format(str(y))
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


filename = base_dir+"MLB_2017/MLB_PitchFX_2017_RegularSeason.csv"
print("Loading test file {}".format(filename))
f2 = pd.read_csv(filename)

test_x = f2[input_labels]
test_y = f2['umpcall']

test_x = test_x.as_matrix()
test_y = test_y.as_matrix()
test_y = vfunc(test_y)

filename = base_dir+"MLB_2017/MLB_PitchFX_2017_PostSeason.csv"
print("Loading test file {}".format(filename))
f2 = pd.read_csv(filename)

tmp_x = f2[input_labels]
tmp_y = f2['umpcall']

tmp_x = tmp_x.as_matrix()
tmp_y = tmp_y.as_matrix()
tmp_y = vfunc(tmp_y)

test_x = np.concatenate((test_x, tmp_x), axis=0)
test_y = np.concatenate((test_y, tmp_y), axis=0)
MAX_GAME_LEN = 597


class Lang:
    def __init__(self, name):
        """Init Lang with a name."""
        self.name = name
        self.word2index = {"<UNK>": 0, '<EMP>':1}
        self.word2count = {}
        self.index2word = {0: "<UNK>", 1:'<EMP>'}
        self.n_words = len(self.word2index)  # Count SOS and EOS

    def addword(self, word):
        """Add a word to the dict."""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

players = Lang('players')
pitchers = Lang('pitchers')
batters = Lang('batters')
loc = Lang('stadium')
dt = Lang('date')
sd = Lang('side')

for i in range(train_x.shape[0]):
    loc.addword(train_x[i,1])    
    sd.addword(train_x[i,3])
    players.addword(train_x[i,4])
    players.addword(train_x[i,5])
    pitchers.addword(train_x[i,4])
    batters.addword(train_x[i,5])

def mat2ind(x):
    def findindex(s,lan):
        try:
            return lan.word2index[s]
        except KeyError:
            return lan.word2index['<UNK>']
    n_train_x = np.zeros((x.shape[0], x.shape[1]+1))
    for i in range(x.shape[0]):
        n_train_x[i,0] = int(x[i,0][5:7]) # month
        n_train_x[i,1] = int(x[i,0][8:]) # day
        n_train_x[i,2] = findindex(x[i,1], loc) # stadium

        n_train_x[i,3] = x[i,2] # inning

        n_train_x[i,4] = 1 if x[i,3]=='top' else -1 # side
        n_train_x[i,5] = findindex(x[i,4], pitchers) # pitcher
        n_train_x[i,6] = findindex(x[i,5], batters) # batter
        #n_train_x[i,7] = findindex('<EMP>', batters) if isinstance(x[i,6],float) and math.isnan(x[i,6]) else findindex(x[i,6], batters)
        #n_train_x[i,8] = findindex('<EMP>', batters)if isinstance(x[i,7],float) and math.isnan(x[i,7]) else findindex(x[i,7], batters)
        #n_train_x[i,9] = findindex('<EMP>', batters) if isinstance(x[i,8],float) and math.isnan(x[i,8]) else findindex(x[i,8], batters)
        n_train_x[i,7] = 0 if isinstance(x[i,6],float) and math.isnan(x[i,6]) else 1
        n_train_x[i,8] = 0 if isinstance(x[i,7],float) and math.isnan(x[i,7]) else 1
        n_train_x[i,9] = 0 if isinstance(x[i,8],float) and math.isnan(x[i,8]) else 1
        n_train_x[i,10] = x[i,9] # pitch_count
        n_train_x[i,11] = x[i,10] # balls
        n_train_x[i,12] = x[i,11] # strikes
    return n_train_x


# In[32]:


nx = mat2ind(train_x)


# In[33]:


ntx = mat2ind(test_x)

def getNextGame(f):
    ctr = 0
    ptr = ctr
    while ctr < f.shape[0]:
        prev_inn = 0
        while ptr < f.shape[0] and f[ctr,0] == f[ptr,0] and f[ctr,1] == f[ptr,1] and f[ctr,2] == f[ptr,2] and f[ptr,3]>=prev_inn :
            prev_inn = f[ptr,3]
            ptr+=1
        yield ctr,ptr
        ctr = ptr

game_ctr = 0
for c,p  in getNextGame(nx):
    game_ctr +=1
print(game_ctr)
ntrain_x = np.zeros((game_ctr, MAX_GAME_LEN, nx.shape[1]))
ntrain_y = np.ones((game_ctr, MAX_GAME_LEN, 1)) * 3
ctr=0
for c,p  in getNextGame(nx):
    ntrain_x[ctr,:p-c,:] = nx[c:p,:]
    ntrain_y[ctr,:p-c,0] = train_y[c:p]
    ctr+=1


# In[36]:


game_ctr = 0
for c,p  in getNextGame(ntx):
    game_ctr +=1
print(game_ctr)
ntest_x = np.zeros((game_ctr, MAX_GAME_LEN, ntx.shape[1]))
ntest_y = np.ones((game_ctr, MAX_GAME_LEN, 1))*3
ctr=0
for c,p  in getNextGame(ntx):
    ntest_x[ctr,:p-c,:] = ntx[c:p,:]
    ntest_y[ctr,:p-c,0] = test_y[c:p]
    ctr+=1

ntrain_x[0]

ntrain_y

DEVICE = torch.device('cuda')

def data_gen(dx, dy, batch_size=100):
    idx = 0
    while True:
        if idx*batch_size >= dx.shape[0]:
            return
        elif (idx+1)*batch_size > dx.shape[0]:
            yield dx[idx*batch_size:,:], dy[idx*batch_size:]
        else:
            yield dx[idx*batch_size:(idx+1)*batch_size,:], dy[idx*batch_size:(idx+1)*batch_size]
        idx += 1


class PlayerEmbedding(nn.Module):
    def __init__(self, inn_dim, pitcher_size, batter_size, pc_dim, balls_dim, strikes_dim , emb_dim):
        super(PlayerEmbedding, self).__init__()
        self.emb_pitchers = nn.Embedding(pitcher_size, emb_dim)
        self.emb_batters = nn.Embedding(batter_size, emb_dim)
        
        self.emb_pc = nn.Embedding(600, pc_dim)
        self.emb_inn = nn.Embedding(20, inn_dim)
        self.emb_balls = nn.Embedding(6, balls_dim)
        self.emb_strikes = nn.Embedding(6, strikes_dim)

        self.emb_dim = balls_dim + strikes_dim + pc_dim + inn_dim + 2*emb_dim
        #pc_dim + self.emb_dim = inn_dim + 

    def forward(self, x):
        e_inn = self.emb_inn(x[:,:,3])
        e_p = self.emb_pitchers(x[:,:,5])
        e_b = self.emb_batters(x[:,:, 6])
        #e_o1 = self.emb_batters(x[:,:,7])
        #e_o2 = self.emb_batters(x[:,:,8])
        #e_o3 = self.emb_batters(x[:,:,9])
        e_pc = self.emb_pc(x[:,:, 10])
        e_bl = self.emb_balls(x[:,:, 11])
        e_st = self.emb_strikes(x[:,:, 12])
        
        

        emb_all = torch.cat([   e_inn,
                                e_p, e_b, 
                                #e_o1, e_o2, e_o3, 
                                e_pc, 
                                e_bl, e_st], dim=2)
        return emb_all

    def init_weights(self):
        initrange = 0.5
        em_layer = [ self.emb_pitchers, self.emb_batters, self.emb_pc, 
                self.emb_inn, 
                self.emb_balls, self.emb_strikes]

        for layer in em_layer:
            #layer.weight.data.normal_(0, initrange)
            layer.weight.data.uniform_(-initrange, initrange)

class LSTM(nn.Module):
    """Vanilla encoder using pure LSTM."""
    def __init__(self, hidden_size, embedding_layer, dp=0.2, n_layers=2):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        #self.embedding = embedding_layer
        #self.dp1 = nn.Dropout(dp)
        self.feat1 = nn.Linear(embedding_layer.emb_dim + 3, hidden_size)
        self.feat2 = nn.Linear(hidden_size, hidden_size)
        self.feat3 = nn.Linear(hidden_size, hidden_size)
        self.feat4 = nn.Linear(hidden_size, hidden_size)
        self.act1 = nn.SELU()
        self.act2 = nn.SELU()
        self.act3 = nn.SELU()
        self.act4 = nn.SELU()
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=dp, bidirectional=False)
        #self.lin1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin2 = nn.Linear(self.hidden_size, 3)

    def forward(self, inp, onb, hidden):
        '''
        inputs: (batch_size, seq_len, feature_dim)
        '''
        #embedded = self.embedding(inp)
        output = inp
        output = torch.cat([output, onb], dim=2)
        output = self.act1(self.feat1(output))
        output = self.act2(self.feat2(output))
        output = self.act3(self.feat3(output))
        output = self.act4(self.feat4(output))
        
        #output = torch.cat([output, onb], dim=2)
        output = output.permute(1,0,2)
        
        bilstm_outs, nh = self.lstm(output, hidden)
        
        output = bilstm_outs.permute(1,0,2)
        # (batch, seq_len, hidden)
        # output = F.relu(self.lin1(output))
        output = self.lin2(output)
        # (batch, seq_len, 3)
        return F.log_softmax(output, dim=2)

    def initHidden(self, batch_size):
        forward = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size )).to(DEVICE)
        backward = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to(DEVICE)
        return (forward, backward)
        #return forward


def train(train_x, train_y, dev_x, dev_y,emb, model, scheduler, optimizer, criterion, batch_size=512, 
          max_epoch = 512, validation_interv=1000, show_iter=10):
    start = time.time()
    best_loss = 1000
    for ep in range(max_epoch):
        print("Epoch {}".format(ep+1))
        model.train()
        #scheduler.step()

        train_iter = data_gen(train_x, train_y, batch_size=batch_size)

        ctr = 0
        avg_loss = 0
        acc = 0
        iteration = 0
        for bx,by in train_iter:
            iteration +=1
            optimizer.zero_grad()
            model.train()
            hid = model.initHidden(bx.shape[0])
            y_pred = model(emb(bx), bx[:,:,7:10].type(torch.FloatTensor).to(DEVICE), hid)
            
            y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1],-1)
            by = by.view(by.shape[0]*by.shape[1])
            
            idx = (by!=3)
            y_pred = y_pred[idx,:]
            by = by[idx]

            _, lab_y = torch.max(y_pred, 1)
            
            
            loss = criterion(y_pred, by)
            acc += torch.sum(lab_y == by).item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) +
                                          list(emb.parameters()), 0.05)
            
            avg_loss += loss.item()*by.shape[0]
            optimizer.step()
            ctr+= by.shape[0]
            if iteration % show_iter == 0:
                print("Time: {}, iter: {}, avg. loss: {}, avg.acc: {}".format(time.time() - start, 
                                                                              iteration,  
                                                                              avg_loss/ctr,
                                                                              acc/ctr))
                avg_loss = 0
                ctr = 0
                acc = 0
        with torch.no_grad():
            lo, ac = calPerf(dev_x, dev_y, emb, model, criterion, batch_size)
            if best_loss > lo:
                best_loss = lo
                torch.save(emb.state_dict(), "best_rnn_emb.pk")
                torch.save(model.state_dict(), "best_rnn_model.pk")

        print("Time: {}, loss:{} dev_loss:{}, dev_acc:{}".format(time.time() - start, avg_loss/ctr, lo, ac))
        import os.path
        if os.path.exists("./STOP"):
            return


def calPerf(dev_x, dev_y, emb, model, criterion, batch_size=16):
    with torch.no_grad():
        model.eval()
        ll = 0
        ctrr = 0
        acc = 0
        for dx,dy in data_gen(dev_x, dev_y, batch_size=batch_size):
            hid = model.initHidden(dx.shape[0])
            dy_pred = model(emb(dx), dx[:,:,7:10].type(torch.FloatTensor).to(DEVICE), hid)

            dy_pred = dy_pred.view(dy_pred.shape[0] * dy_pred.shape[1], -1)
            dy = dy.view(dy.shape[0]*dy.shape[1])

            idx = (dy!=3)
            dy_pred = dy_pred[idx,:]
            dy = dy[idx]

            loss = criterion(dy_pred, dy)

            ll += loss * dy.shape[0]
            ctrr += dy.shape[0]
            _, lab_y = torch.max(dy_pred, 1)

            acc += torch.sum( lab_y == dy ).item()
    return ll/ctrr, acc/ctrr


def calPred(dev_x, dev_y, emb, model, criterion, batch_size=16):
    with torch.no_grad():
        model.eval()
        ll = 0
        ctrr = 0
        acc = 0
        pred_y = []
        for dx,dy in data_gen(dev_x, dev_y, batch_size=batch_size):
            hid = model.initHidden(dx.shape[0])
            dy_pred = model(emb(dx), dx[:,:,7:10].type(torch.FloatTensor).to(DEVICE), hid)

            dy_pred = dy_pred.view(dy_pred.shape[0] * dy_pred.shape[1], -1)
            dy = dy.view(dy.shape[0]*dy.shape[1])

            idx = (dy!=3)
            dy_pred = dy_pred[idx,:]
            dy = dy[idx]
            pred_y.append(dy_pred)

            loss = criterion(dy_pred, dy)

            ll += loss * dy.shape[0]
            ctrr += dy.shape[0]
            _, lab_y = torch.max(dy_pred, 1)

            acc += torch.sum( lab_y == dy ).item()
        pred_y = torch.cat(pred_y, dim=0)
    return pred_y

vtrue_y = Variable(torch.from_numpy(test_y.astype(np.long))).to(DEVICE)


from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import pickle
import seaborn as sn
import matplotlib.pyplot as plt

def generate_confusion_matrix(true_y, pred_y, file_name = "confusion_matrix.png"):
    ax= plt.subplot()
    cm = confusion_matrix(true_y, pred_y)
    df_cm = pd.DataFrame(cm, range(3), range(3))
    sn.set(font_scale=1.2)#for label size
    sn.heatmap(df_cm, annot=True, fmt="d",annot_kws={"size": 14}, cmap="YlGnBu")
#     ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['Ball', 'Strike', 'Hit']) 
    ax.yaxis.set_ticklabels(['Ball', 'Strike', 'Hit'])
    plt.savefig(file_name)
    #plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Seq model')
    parser.add_argument('--best', action='store_true', help='use best model')
    parser.add_argument('--eval', action='store_true', help='no training model')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--maxepoch', type=int, default=100)
    args = parser.parse_args()
    print(args)
    
    from sklearn.model_selection import train_test_split
    train_x, dev_x, train_y, dev_y = train_test_split(ntrain_x, ntrain_y, test_size=0.1, random_state=args.seed, shuffle=True)
    vtrainx = Variable(torch.from_numpy(train_x.astype(np.long)), requires_grad=False).to(DEVICE)
    vtrainy = Variable(torch.from_numpy(train_y.astype(np.long)), requires_grad=False).to(DEVICE)

    vdevx = Variable(torch.from_numpy(dev_x.astype(np.long)), requires_grad=False).to(DEVICE)
    vdevy = Variable(torch.from_numpy(dev_y.astype(np.long)), requires_grad=False).to(DEVICE)
    vtestx = Variable(torch.from_numpy(ntest_x.astype(np.long)), requires_grad=False).to(DEVICE)
    vtesty = Variable(torch.from_numpy(ntest_y.astype(np.long)), requires_grad=False).to(DEVICE)

    emb = PlayerEmbedding(5, pitchers.n_words, batters.n_words, 5, 5, 5, 10).to(DEVICE)


    model = LSTM(200, emb, dp=0.1, n_layers=1).to(DEVICE)
    print(model)
    opt = torch.optim.Adam(list(model.parameters()) + list(emb.parameters()), lr=1e-3)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5, eta_min=0)
    sched = None
    # opt = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
    crit = nn.NLLLoss()

    emb.init_weights()
    if args.best:
        print("load best model")
        model.load_state_dict(torch.load("best_rnn_model.pk"))
        emb.load_state_dict(torch.load("best_rnn_emb.pk"))
    if not args.eval:
        train(vtrainx, vtrainy, vdevx, vdevy,emb, model, sched, opt, crit, batch_size=512, max_epoch=args.maxepoch, show_iter=10)
    model.load_state_dict(torch.load("best_rnn_model.pk"))
    emb.load_state_dict(torch.load("best_rnn_emb.pk"))
    py = calPred(vtestx, vtesty, emb, model, crit)
    test_loss = crit(py, vtrue_y).item()
    _, label_y = torch.max(py, dim=1)
    test_acc = torch.sum(label_y == vtrue_y).item() / vtrue_y.shape[0]
    label_y = label_y.cpu().numpy()
    torch.save(emb.state_dict(), "rnn_emb.pk")
    torch.save(model.state_dict(), "rnn_model.pk")
    print("test loss: {}, test_acc: {}".format(test_loss, test_acc))
    generate_confusion_matrix(test_y, label_y, "RNN_confusion.png")

