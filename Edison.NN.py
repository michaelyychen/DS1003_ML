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
total_row, n_features = x_train.shape

label2one = {'B':0,'S':1,'X':2}
one2label = {0:'B', 1:'S', 2:'X'}
vfunc = np.vectorize(lambda x:label2one[x])

y_train = vfunc(y_train)
y_test = vfunc(y_test)

class ResNet(nn.Module):
    def __init__(self, in_dim):
        super(ResNet, self).__init__()

        self.lin1 = nn.Linear(in_dim, in_dim)
        self.lin2 = nn.Linear(in_dim, in_dim)
    def forward(self, x):
        output = F.relu(self.lin1(x))
        return self.lin2(output) + x

class Fcc(nn.Module):
    def __init__(self, D_in):
        super(Fcc, self).__init__()
        p = 0.3
        res_dim=1024

        self.lin1 = nn.Linear(D_in, 4096)
        #self.lin1 = nn.Linear(2*self.embedding.embedding_dim, hid_dim[0])
        #self.drop2 = nn.Dropout(p)
        self.res2 = ResNet(4096)
        # self.drop3 = nn.Dropout(p)
        self.res3 = ResNet(4096)
        #self.drop4 = nn.Dropout(p)

        self.lin3 = nn.Linear(4096, 2048)

        self.res4 = ResNet(2048)
        # self.drop5 = nn.Dropout(p)
        self.res5 = ResNet(2048)
        #self.drop6 = nn.Dropout(p)
        self.lin6 = nn.Linear(2048, 1024)
        # self.res6 = ResNet(1024)
        # self.drop5 = nn.Dropout(p)
        self.res7 = ResNet(1024)
        self.lin7 = nn.Linear(1024,512)
        self.res8 = ResNet(512)
        self.lin8 = nn.Linear(512,3)
    
    def forward(self, x):

        #output = self.drop1(output)
        output = self.lin1(x)
        output = F.relu(output)
        
        #output = self.drop2(output)
        output = self.res2(output)
        output = F.relu(output)
        
        # output = self.drop3(output)
        output = self.res3(output)
        output = F.relu(output)
        
        output = F.relu(self.lin3(output))

        #output = self.drop4(output)
        output = self.res4(output)
        output = F.relu(output)

        # output = self.drop5(output)
        output = self.res5(output)
        output = F.relu(output)

        output = self.lin6(output)
        output = F.relu(output)

        # output = self.res6(output)
        # output = F.relu(output)
        # output = self.drop5(output)
        output = self.res7(output)
        output = F.relu(output)

        output = self.lin7(output)
        output = F.relu(output)

        output = self.res8(output)
        output = F.relu(output)

        output = self.lin8(output)
        output = F.log_softmax(output, dim=1)

        return output

vdx = Variable(torch.from_numpy(x_test.A.astype(np.float32))).cuda()
vdy = Variable(torch.from_numpy(y_test)).long().cuda()
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

def train(x_train, y_train, vdx, vdy, model, optimizer, criterion, batch_size=512, max_epoch = 512, validation_interv=1000):
    for ep in range(max_epoch):
        vtx = Variable(torch.from_numpy(x_train.A.astype(np.float32)), requires_grad=False).cuda()
        vty = Variable(torch.from_numpy(y_train), requires_grad=False).long().cuda()
        print("Epoch {}".format(ep))
        train_iter = data_gen(vtx, vty, batch_size=batch_size)
        ctr = 1
        avg_loss = 0
        for bx,by in train_iter:
            optimizer.zero_grad()
            model.train()
            y_pred = model(bx)
#             print(y_pred, by)
            loss = criterion(y_pred, by)
            loss.backward()
            avg_loss += loss.data[0]
            optimizer.step()
            if ctr%validation_interv==0:
                model.eval()
                ll = 0
                ctrr = 0
                for dx,dy in data_gen(vdx, vdy, batch_size=batch_size):
                    dy_pred = model(dx)
                    tmp = criterion(dy_pred, dy).data[0]
                    ll += tmp * dx.shape[0]
                    ctrr += dx.shape[0]
                print("loss:{} dev_loss:{}".format(avg_loss/ctr, ll/ctrr))
            ctr+=1
        del vtx, vty,train_iter

bt_size = 4096
model = Fcc(n_features).cuda()
print(model)
opt = torch.optim.SGD(model.parameters(), lr=1e-3,momentum=0.9)
# opt = torch.optim.Adam(model.parameters(), lr=3e-3)
crit = nn.NLLLoss()
train(x_train, y_train, vdx, vdy, model, opt, crit, batch_size=bt_size, max_epoch=30, validation_interv=100)

model.eval()
ll=0
ctrr=0
for dx,dy in data_gen(vdx, vdy, batch_size=bt_size):
    dy_pred = model(dx)
    pred = dy_pred.exp().cpu().data.numpy()
    y_pred = np.argmax(pred, axis=1)
    tmp = np.sum( y_pred == dy.cpu().data.numpy())
    ll += tmp
    ctrr += dx.shape[0]

print(ll/ctrr)