import pandas as pd
import numpy as np
import pickle
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import KFold, LeaveOneGroupOut, GroupKFold
import warnings
warnings.filterwarnings('once')
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'

from scipy.special import softmax
from sklearn import metrics

from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut, GroupKFold
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import r2_score, make_scorer, roc_auc_score, precision_recall_curve, auc
from scipy.stats import pearsonr, sem, spearmanr
import xgboost
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error


def get_rmse(x,y):
    return np.sqrt((np.mean((x-y)**2)))


def get_pearson(x,y):
    return pearsonr(x,y)[0]

def get_spearman(x,y):
    return spearmanr(x,y)[0]

def get_auc_roc(y,y_pred):
    true_bool = np.array([i > 0.0 for i in y])
    scores = np.array(y_pred)
    auc = roc_auc_score(true_bool, scores)
    return auc

def get_auc_prc(exp, calc):
    exp= np.array(exp)
    exp[exp >=0.25] =1
    exp[exp<=-0.25] =0 
    exp[(exp>-0.25) & (exp < 0.25)]=-1
    #true_bool = np.array([x >= 0.25 or x <= -0.25 for x in exp]) # identify resistant mutations (ddG > 1.36 kcal/mol)
    scores = np.array(calc) # the scores are the calc ddg (higher ddg = higher probability of resistance)
    precision, recall, thresholds = precision_recall_curve(exp, scores, pos_label=1)
    auc_score = auc(recall, precision)
    return auc_score

class MyDataset(Dataset):
    def __init__(self, X1, X2, X3, Y):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.Y = Y
        
    def __len__(self): 
        return self.X1.shape[0]
    
    def __getitem__(self, idx):
        x1 = self.X1[idx]
        x2 = self.X2[idx]
        x3 = self.X3[idx]
        y = self.Y[idx]
        return x1,x2,x3,y

class AugSSP(nn.Module):
    def __init__(self, drug_dim, gene_dim, device):
        super(AugSSP, self).__init__()
        self.drug_dim = drug_dim
        self.gene_dim = gene_dim
        self.device = device
        self.processA = nn.Sequential(nn.Linear(self.drug_dim, 1024),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 256),
                                      nn.ReLU(inplace=True))
        self.processB = nn.Sequential(nn.Linear(self.drug_dim, 1024),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 256),
                                      nn.ReLU(inplace=True))        
        self.processG = nn.Sequential(
            nn.Linear(self.gene_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True))

        self.merged1 = nn.Sequential(
            nn.Linear(256*3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Hardtanh(-1,1))
        
        self.merged2 = nn.Sequential(
            nn.Linear(256*3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Hardtanh(-1,1))

    def forward(self, drugA, drugB, gene):

        xA = self.processA(drugA)
        xB = self.processB(drugB)
        xG = self.processG(gene.float())
        
        x1 = torch.cat((xA, xB, xG), dim = 1)
        x2 = torch.cat((xB, xA, xG), dim = 1)
        out1 = self.merged1(x1)
        out2 = self.merged2(x2)
        m = torch.mean(torch.stack([out1,out2], dim=0), dim=0)
        return m, out1, out2, xA, xB, xG

class trainer(object):
    def __init__(self, dt, model, n_splits = 5, lr = 0.0001, path = '.',
                 batch_size = 64, device = 'cuda:1', alpha = 1,):
        self.dt = dt
        self.n_splits = n_splits
        self.path = path
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.kfold = self.kfold_split()
        self.best = 0
        self.alpha = alpha
        
    def kfold_split(self):
        kfold = list(GroupKFold(n_splits=self.n_splits).split(X=self.dt[0], 
                                                  groups = self.dt[0].index))
        return kfold
                        
    def one_fold_train(self, epoch = 100):
        self.kfold = self.kfold_split()
        i_train, i_test = self.kfold[0]
        dt_train = MyDataset(self.dt[0].iloc[i_train,:].values,
                             self.dt[1].iloc[i_train,:].values,
                             self.dt[2].iloc[i_train,:].values,
                             self.dt[3].iloc[i_train].values)
        dt_test = MyDataset(self.dt[0].iloc[i_test,:].values, 
                            self.dt[1].iloc[i_test,:].values,
                            self.dt[2].iloc[i_test,:].values,
                            self.dt[3].iloc[i_test].values)
        dl_train = DataLoader(dt_train, batch_size=self.batch_size, shuffle=True)
        dl_test = DataLoader(dt_test, batch_size=self.batch_size, shuffle=True)
        self.train(dl_train, dl_test, epoch)

            
    def train(self, dl_train, dl_test, epoch = 100):
        for i in range(epoch):
            self.model.train()
            train_loss = 0
            for x1,x2,x3, y in dl_train:
                
                x1 = x1.clone().detach().float().to(self.device)
                x2 = x2.clone().detach().float().to(self.device)
                x3 = x3.clone().detach().float().to(self.device)
                y = y.clone().detach().float().to(self.device)
                
                self.optimizer.zero_grad()
                m, o1, o2, _, _, _ = self.model(x1, x2, x3)
                loss_m = self.criterion(m, y.reshape(-1,1))
                s = o1 - o2
                loss_s = self.criterion(s.abs(), torch.zeros(y.shape).to(self.device))
                loss = loss_m + self.alpha*loss_s
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
        
            with torch.no_grad():
                val_loss = 0
                self.model.eval()
                for x1,x2,x3,y in dl_test:
                    x1 = x1.clone().detach().float().to(self.device)
                    x2 = x2.clone().detach().float().to(self.device)
                    x3 = x3.clone().detach().float().to(self.device)
                    y_pred, _, _, _, _, _ = self.model(x1, x2, x3)
                    y_pred = y_pred.clone().detach().cpu().numpy()                    
                    y = np.array(y)
                    pear = get_pearson(y, y_pred)
                    spear = get_spearman(y, y_pred)
                    auc = get_auc_roc(y, y_pred)
                    prc = get_auc_prc(y, y_pred)

            print('[EP%s] [loss] %f  [pear] %f  [spear] %f  [auc] %f  [prc] %f'%(str(i).zfill(3), train_loss, pear, spear,
                                                              auc, prc))
            
            if i> 20 and pear > self.best:
                self.best = pear
                self.save_model()
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.path+'model.pkl')
            
    def load_model(self):
        self.model.load_state_dict(torch.load(self.path+'model.pkl', 
                                                map_location=self.device))
    
    def predict(self, dt):
        m = [x for x in range(dt[0].shape[0])]
        a = [m[x:x+self.batch_size] for x in range(0, dt[0].shape[0], self.batch_size)]

        pbar = tqdm(range(len(a)), position = 0, leave = True)
        pred = []
        for i in pbar:
            idx = a[i]
            x1 = dt[0].iloc[idx, :].values
            x2 = dt[1].iloc[idx, :].values
            x3 = dt[2].iloc[idx, :].values
            x1 = torch.tensor(x1).float().to(self.device)
            x2 = torch.tensor(x2).float().to(self.device)
            x3 = torch.tensor(x3).float().to(self.device)
            out,_,_,_,_,_ = self.model(x1, x2, x3) 
            out = out.cpu().detach().numpy()
            pred.append(out)
        return pred



























