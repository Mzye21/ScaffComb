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
    true_bool = np.array([x >= 0.25 or x <= -0.25 for x in exp]) # identify resistant mutations (ddG > 1.36 kcal/mol)
    scores = np.array(calc) # the scores are the calc ddg (higher ddg = higher probability of resistance)
    precision, recall, thresholds = precision_recall_curve(true_bool, scores)
    auc_score = auc(recall, precision)
    return auc_score

class SSbalanced(nn.Module):
    def __init__(self, gene_dim, drug_dim, latent_dim, 
                 token_dim = 20, n_lstm_layers = 3, device = 'cuda:0'):
        super(SSbalanced, self).__init__()
        self.gene_dim = gene_dim
        self.drug_dim = drug_dim
        self.latent_dim = latent_dim 
        self.token_dim = token_dim
        self.n_lstm_layers = n_lstm_layers
        self.device = device
        
        self.embedA = nn.Embedding(self.token_dim, self.latent_dim)
        self.lstmA = nn.LSTM(self.latent_dim, 
                             self.latent_dim, 
                             self.n_lstm_layers,
                             batch_first = True,
                             bidirectional = True,
                             dropout = 0.2)
        self.processA = nn.Sequential(nn.Linear(self.drug_dim*self.latent_dim, 1024),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 256),
                                      nn.ReLU(inplace=True))
        
        self.embedB = nn.Embedding(self.token_dim, self.latent_dim)
        self.lstmB = nn.LSTM(self.latent_dim, 
                             self.latent_dim, 
                             self.n_lstm_layers,
                             batch_first = True,
                             bidirectional = True,
                             dropout=0.2)
        self.processB = nn.Sequential(nn.Linear(self.drug_dim*self.latent_dim, 1024),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 256),
                                      nn.ReLU(inplace=True))
        
        self.embed_G = nn.Sequential(
            nn.Linear(978, 512),
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

    def forward(self, drugA, drugB, gene, seq_A, seq_B):
        
        batch_size = drugA.shape[0]
        
        hidden = (self.init_hidden(batch_size), 
                  self.init_cell(batch_size))
        
        xA = self.embedA(drugA)
        xA = pack_padded_sequence(xA, seq_A,
                                 batch_first=True,
                                 enforce_sorted=False)
        xA, _ = self.lstmA(xA, hidden)
        xA, _ = pad_packed_sequence(xA, batch_first= True, 
                                   total_length=self.drug_dim)
        xA = xA.view(batch_size,-1,2,self.latent_dim).sum(dim =2).squeeze()
        xA = self.processA(xA.view(batch_size, -1))

        
        xB = self.embedB(drugB)
        xB = pack_padded_sequence(xB, seq_B,
                                 batch_first=True,
                                 enforce_sorted=False)
        xB, _ = self.lstmB(xB, hidden)
        xB, _ = pad_packed_sequence(xB, batch_first= True, 
                                   total_length=self.drug_dim)
        xB = xB.view(batch_size,-1,2,self.latent_dim).sum(dim =2).squeeze()
        xB = self.processB(xB.view(batch_size, -1))

        xG = self.embed_G(gene.float())
        
        x1 = torch.cat((xA, xB, xG), dim = 1)
        x2 = torch.cat((xB, xA, xG), dim = 1)
        out1 = self.merged1(x1)
        out2 = self.merged2(x2)
        
        m = torch.mean(torch.stack([out1,out2], dim=0), dim=0)
        
        return m, out1, out2, xA, xB, xG
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.n_lstm_layers*2, batch_size, self.latent_dim).to(self.device)

    def init_cell(self, batch_size):
        return torch.zeros(self.n_lstm_layers*2, batch_size, self.latent_dim).to(self.device)

class trainer(object):
    def __init__(self, tokens, dt_train = None, dt_val = None,
                 gene_dim = 978, drug_dim =100, latent_dim =512, 
                 batch_size = 32,  lr = 0.0001, alpha = 1, 
                 device='cuda:1', path = '.'):
        super(trainer, self).__init__()
        
        self.gene_dim = gene_dim
        self.drug_dim = drug_dim
        self.latent_dim = latent_dim
        self.tokens = tokens + ['6']
        self.token_map = {self.tokens[i]:i for i in range(len(self.tokens))}
        self.rev_map = {v:k for k,v in self.token_map.items()}
        #处理长token
        self.token_map['A'] = self.token_map['Br']
        self.token_map['D'] = self.token_map['Cl']
        self.token_map.pop('Br')
        self.token_map.pop('Cl')
        self.n_tokens = len(self.tokens)
        self.dt_train = dt_train
        self.dt_val = dt_val

        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha        
        self.device = device
        
        self.model = SSbalanced(self.gene_dim, 
                                self.drug_dim, 
                                self.latent_dim, 
                                self.n_tokens, 
                                n_lstm_layers=2,
                                device=self.device).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.path = path
        
    def Smi2Tensor(self, smis):
        '''将smi转化为tensor'''
        out = []
        long_token = {'Br':'A', 'Cl':'D'}
        #if len(smis) > 200:
        #    pbar = tqdm(range(len(smis)), position=0, leave=True)
        #    pbar.set_description('Smi2Tensor')
        #else:
        pbar = range(len(smis))
        for i in pbar:
            smi = smis[i]
            #将smi中的长token转换为短token
            for j in long_token.keys():
                smi = smi.replace(j, long_token[j])
            tensor = torch.zeros(len(smi)).long()
            for j in range(len(smi)):
                if smi[j] in self.token_map.keys():
                    tensor[j] = self.token_map[smi[j]]
                else:
                    print(smi, smi[j])
            out.append(tensor)
        return out        

    def SeqPaddding(self, t, padding = 0):
        padded = torch.LongTensor(np.zeros((len(t), self.drug_dim)).astype(int))
        seq_length = []
        for i in range(len(t)):
            length = t[i].shape[0]
            s = torch.LongTensor(np.zeros((self.drug_dim)).astype(int))
            s[:length] = t[i]
            padded[i] = s
            seq_length.append(length)
        seq_length = torch.LongTensor(seq_length).to(self.device)
        return padded, seq_length 

    
    def train(self, end, start = 0):
        best = 0
        epochs = end+start
        for epoch in range(start, epochs):
            self.model.train()
            
            dt_e = [self.sample_a_batch(self.dt_train) for _ in range(10)]
            
            training_loss = 0
            for i in range(len(dt_e)):
                dA_batch_0, dB_batch_0, gene_batch_0, ss_batch_0, seq_A_batch_0, seq_B_batch_0 = dt_e[i]
                                
                #加入反向样本
                dA_batch = torch.cat([dA_batch_0, dB_batch_0], dim=0)
                dB_batch = torch.cat([dB_batch_0, dA_batch_0], dim=0)
                seq_A_batch = torch.cat([seq_A_batch_0, seq_B_batch_0], dim=0)
                seq_B_batch = torch.cat([seq_B_batch_0, seq_A_batch_0], dim=0)
                gene_batch = torch.cat([gene_batch_0, gene_batch_0], dim=0)
                ss_batch = torch.cat([ss_batch_0, ss_batch_0], dim=0)
                                
                self.optimizer.zero_grad()
                
                m, o1, o2, _, _, _ = self.model(dA_batch.to(self.device), 
                                       dB_batch.to(self.device), 
                                       gene_batch.float().to(self.device),
                                       seq_A_batch.to(self.device),
                                       seq_B_batch.to(self.device))
                
                loss_m = self.criterion(m, ss_batch.float().to(self.device))
                s = o1 - o2
                loss_s = self.criterion(s.abs(), torch.zeros(ss_batch.shape).to(self.device))
                                
                loss = loss_m + self.alpha*loss_s
                
                loss.backward() 
                self.optimizer.step()
                training_loss += loss.item()
                
            training_loss = training_loss/len(dt_e)
            
            # 验证
            pear, spear, auc, prc = self.eval(epoch, self.dt_val)
            print('[EP%s] [loss] %f  [pear] %f  [spear] %f  [auc] %f  [prc] %f'%(str(epoch).zfill(3), training_loss, pear, spear,
                                                              auc, prc))
            
            if pear > best:
                self.save_model()
                best = pear
            
    def sample_a_batch(self, dt):
        sample_batch = random.sample(range(dt[0].shape[0]), self.batch_size)
        drugA = self.Smi2Tensor(dt[0].iloc[sample_batch, 0].values)
        drugB = self.Smi2Tensor(dt[0].iloc[sample_batch, 1].values)
        drugA, seq_len_A = self.SeqPaddding(drugA)
        drugB, seq_len_B = self.SeqPaddding(drugB) 
        gene_exp =  torch.tensor(dt[1].iloc[sample_batch,:].values)
        ss =  torch.tensor(dt[2].iloc[sample_batch].values)
        ss = ss/20
        ss[ss > 1] = 1
        ss[ss < -1] = -1
        return drugA, drugB, gene_exp, ss, seq_len_A, seq_len_B
        
    def eval(self, epoch, dt_e=None):
        self.model.eval()
        dt_e = [self.sample_a_batch(dt_e) for _ in range(10)]
        for i in range(len(dt_e)):
            dA_batch, dB_batch, gene_batch, ss_batch, seq_A_batch, seq_B_batch  = dt_e[i]
            m, _, _, _, _, _ = self.model(dA_batch.to(self.device), 
                                   dB_batch.to(self.device), 
                                   gene_batch.float().to(self.device),
                                   seq_A_batch.to(self.device),
                                   seq_B_batch.to(self.device))
            m = m.cpu().detach().numpy()
            pear = get_pearson(ss_batch, m)
            spear = get_spearman(ss_batch, m)
            auc = get_auc_roc(ss_batch, m)
            prc = get_auc_prc(ss_batch, m)
        return pear, spear, auc, prc                    

    def save_model(self):
        torch.save(self.model.state_dict(), self.path+'model.pkl')
            
    def load_model(self):
        self.model.load_state_dict(torch.load(self.path+'model.pkl', 
                                                map_location=self.device))

def SDSP_predict(md, dt):
    dt_e = []
    for _ in range(10):
        sample_batch = random.sample(range(dt[0].shape[0]), 100)
        drugA = md.Smi2Tensor(dt[0].iloc[sample_batch,0].tolist())
        drugB = md.Smi2Tensor(dt[1].iloc[sample_batch,0].tolist())
        drugA, seq_len_A = md.SeqPaddding(drugA)
        drugB, seq_len_B = md.SeqPaddding(drugB) 
        gene_exp =  torch.tensor(dt[2].iloc[sample_batch,:].values)
        dt_e.append([drugA, drugB, gene_exp, seq_len_A, seq_len_B])
    md.model.eval()
    pred = []
    pred_rev = []
    with torch.no_grad():
        for i in range(len(dt_e)):
            dA_batch, dB_batch, gene_batch, seq_A_batch, seq_B_batch  = dt_e[i]
            m, _, _, _, _, _ = md.model(dA_batch.to(device), 
                                   dB_batch.to(device), 
                                   gene_batch.float().to(device),
                                   seq_A_batch.to(device),
                                   seq_B_batch.to(device))
            m = m.cpu().detach().numpy()
            m_, _, _, _, _, _ = md.model(dB_batch.to(device), 
                                   dA_batch.to(device), 
                                   gene_batch.float().to(device),
                                   seq_B_batch.to(device),
                                   seq_A_batch.to(device))
            m_ = m_.cpu().detach().numpy()
            pred.append(m)
            pred_rev.append(m_)
    return pred, pred_rev
