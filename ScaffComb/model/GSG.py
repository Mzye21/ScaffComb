import pandas as pd
import numpy as np
import pickle
import random
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm
sns.set_style('white')
from collections import Counter
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Recap
from utils import MolSplit
from rdkit import DataStructs
from rdkit.Chem import AllChem
from Levenshtein import distance as leven_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable
import warnings
warnings.filterwarnings('once')
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
from scipy.special import softmax
from sklearn import metrics

class GeneEncoder(nn.Module):
    def __init__(self, gene_dim, hidden_dim,
		 context_dim = 40,
                 device = 'cuda:0'):
        super(GeneEncoder, self).__init__()
        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        self.device = device
	self.context_dim = context_dim
        
        self.dense = nn.Sequential(nn.Linear(self.gene_dim, 4096),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(4096, self.context_dim*self.hidden_dim),
                                   nn.ReLU())
    def forward(self, inpt):
        output = self.dense(inpt).view(-1, self.context_dim, self.hidden_dim)
        return output

class StackAttnLSTM(nn.Module):
    def __init__(self, input_dim, gene_dim, hidden_dim, out_size,
                 n_lstm_layer = 1, is_bi = False, context_dim = 40,
                 stack_width = 1500, stack_depth = 200, 
                 device = 'cuda:0'):
        super(StackAttnLSTM, self).__init__()
        self.input_dim = input_dim
        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        self.out_size = out_size
        self.n_lstm_layer = n_lstm_layer
        self.stack_width = stack_width
        self.stack_depth = stack_depth
        self.device = device
        self.is_bi = is_bi
	self.context_dim = context_dim
        if is_bi:
            self.n_di = 2
        else:
            self.n_di = 1
        
        # the attention
        self.attn = nn.Linear(self.hidden_dim * 2, self.context_dim)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # the stack
        self.stack_input = nn.Linear(self.hidden_dim*self.n_di, self.stack_width)
        self.stack_ctrl_layer = nn.Linear(self.hidden_dim*self.n_di, 3)
        
        # the lstm
        self.encoder = nn.Embedding(self.input_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(self.hidden_dim + self.stack_width, self.hidden_dim,
                           self.n_lstm_layer, bidirectional = self.is_bi)
        self.decoder = nn.Linear(self.hidden_dim*self.n_di, self.out_size)
        
    def forward(self, inpt, hidden, stack, encoder_outputs = None):
        # if no gene
        if type(encoder_outputs) == type(None):
            return self.forward_uncond(inpt, hidden, stack)
        # if given gene
        else:
            return self.forward_cond(inpt, hidden, stack, encoder_outputs)
        
    def forward_uncond(self, inpt, hidden, stack):       
        inpt = self.encoder(inpt.view(1,-1))
        hidden_ = hidden[0]
                
        if self.is_bi:
            hidden_2_stack = torch.cat((hidden_[0], hidden_[1]), dim = 1)                
        else:
            hidden_2_stack = hidden_[0]
        
        stack_ctrl = self.stack_ctrl_layer(hidden_2_stack)
        stack_ctrl = F.softmax(stack_ctrl, dim = 1)        
        
        stack_input = self.stack_input(hidden_2_stack.unsqueeze(0))
        stack_input = torch.tanh(stack_input)
        stack = self.stack_augmentation(stack_input.permute(1,0,2),
                                       stack,
                                       stack_ctrl)
        stack_top = stack[:,0,:].unsqueeze(0)
        inpt = torch.cat((inpt, stack_top), dim = 2)        
        output0, next_hidden = self.lstm(inpt.view(1, 1, -1), hidden)
        output = self.decoder(output0.view(1, -1))
        return output, next_hidden, stack

    def forward_cond(self, inpt, hidden, stack, encoder_outputs):       
        inpt = self.encoder(inpt.view(1,-1))
        
        embedded = self.dropout(inpt.view(1,1,-1))
        
        hidden_ = hidden[0]
        
        # stack part
        if self.is_bi:
            hidden_2_stack = torch.cat((hidden_[0], hidden_[1]), dim = 1)                
        else:
            hidden_2_stack = hidden_[0]
        
        stack_ctrl = self.stack_ctrl_layer(hidden_2_stack)
        stack_ctrl = F.softmax(stack_ctrl, dim = 1)        
        
        stack_input = self.stack_input(hidden_2_stack.unsqueeze(0))
        stack_input = torch.tanh(stack_input)
        stack = self.stack_augmentation(stack_input.permute(1,0,2),
                                       stack,
                                       stack_ctrl)
        stack_top = stack[:,0,:].unsqueeze(0)
        
        attn_weights = torch.softmax(self.attn(torch.cat((embedded[0], hidden_[0]), 1)), 
                                 dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0)) #(1,1, hidden_dim)
        attn_output = torch.cat((embedded[0], attn_applied[0]), 1)
        attn_output = self.attn_combine(attn_output).unsqueeze(0)
        attn_output = torch.tanh(attn_output)
        
        inpt_lstm = torch.cat((attn_output, stack_top), dim = 2)        
        output0, next_hidden = self.lstm(inpt_lstm.view(1, 1, -1), hidden)
        output = self.decoder(output0.view(1, -1))
        return output, next_hidden, stack, attn_weights

    def stack_augmentation(self, inpt_val, prev_stack, ctrls):
        batch_size = prev_stack.size(0)  
        ctrls = ctrls.view(-1,3,1,1)
        zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width).to(self.device)
        zeros_at_the_bottom = Variable(zeros_at_the_bottom)
        a_push, a_pop, a_no_op = ctrls[:,0], ctrls[:,1], ctrls[:, 2]
        stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim = 1)
        stack_up  = torch.cat((inpt_val, prev_stack[:, :-1]), dim = 1)
        new_stack = a_no_op*prev_stack + a_pop*stack_down + a_push*stack_up
        return new_stack
    
    def init_hidden(self):
        return torch.zeros(self.n_lstm_layer*self.n_di, 1, self.hidden_dim).to(self.device)

    def init_cell(self):
        return torch.zeros(self.n_lstm_layer*self.n_di, 1, self.hidden_dim).to(self.device)

    def init_stack(self):
        return torch.zeros(1, self.stack_depth, self.stack_width).to(self.device)

class AttnDGenerator(object):
    def __init__(self, tokens, dt = None, 
                 n_lstm_layer = 1, gene_dim=978, hidden_dim = 1500, context_dim = 40, is_bi = False, 
                 stack_depth = 200, stack_width = 1500, batch_size = 128,
                 lr = 0.001, device = 'cuda:0', path='./'):
        super(AttnDGenerator,self).__init__()
        
        self.batch_size = batch_size
        self.tokens = ['<','>','*','6'] + tokens
        self.token_map = {self.tokens[i]:i for i in range(len(self.tokens))}
        self.rev_map = {v:k for k,v in self.token_map.items()}
        self.token_map['A'] = self.token_map['Br']
        self.token_map['D'] = self.token_map['Cl']
        self.token_map.pop('Br')
        self.token_map.pop('Cl')
        self.n_tokens = len(self.tokens)
        
        self.lr = lr
        self.device = device
        self.path = path
        self.logs = SummaryWriter(self.path+'runs/')
        
        print('processing smile seqs...')
        if dt != None:
            self.smis = dt[3]
            self.gene = dt[1]
            self.smis = ['<'+x+'>' for x in self.smis]
        
        self.n_lstm_layer = n_lstm_layer
        self.hidden_dim = hidden_dim
        self.gene_dim = gene_dim
	self.context_dim = context_dim
        self.stack_depth = stack_depth
        self.stack_width = stack_width
        self.is_bi = is_bi
        if is_bi:
            self.n_bi = 2
        else:
            self.n_bi = 1
            
        print('building model...')
        self.encoder = GeneEncoder(self.gene_dim, self.hidden_dim,context_dim = self.context_dim,
                                   device = self.device).to(self.device)
        self.decoder = StackAttnLSTM(self.n_tokens, self.gene_dim, self.hidden_dim, self.n_tokens,
                             n_lstm_layer = self.n_lstm_layer, is_bi = self.is_bi, context_dim = self.context_dim,
                             stack_width = self.stack_width, stack_depth = self.stack_depth,
                             device = self.device).to(self.device)
        self.init_hidden = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.init_cell = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)

        print('adding graph...')
        # optimizer
        self.encoder_optimizer = optim.RMSprop(self.encoder.parameters(), 
                                               lr = self.lr)
        self.decoder_optimizer = optim.RMSprop(self.decoder.parameters(), 
                                               lr = self.lr)
        # loss
        self.criterion = nn.CrossEntropyLoss().to(self.device)
    
    def change_lr(self, new_lr):
        self.lr = new_lr
        self.encoder_optimizer = optim.RMSprop(self.encoder.parameters(), 
                                               lr = self.lr)
        self.decoder_optimizer = optim.RMSprop(self.decoder.parameters(), 
                                               lr = self.lr)
        
    def ED_train_one_step(self, instr, tstr, gene):
        
        self.encoder.train()
        self.decoder.train()
        
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
            
        encoder_outputs = self.encoder(gene)        
        decoder_hidden = (self.init_hidden(encoder_outputs.mean(dim=1)).view(1,1,-1), 
                          self.init_cell(encoder_outputs.mean(dim=1)).view(1,1,-1))
        
        stack = self.decoder.init_stack()
        loss = 0
        for i in range(len(instr)):
            decoder_output, decoder_hidden, stack, attn_weight = self.decoder(instr[i], 
                                                         decoder_hidden,
                                                         stack,
                                                         encoder_outputs.squeeze(0))
            loss_s = self.criterion(decoder_output, tstr[i].unsqueeze(0))
            loss+= loss_s
            
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.item()/len(instr)

    def D_train_one_step(self, instr, tstr):
        self.decoder.train()
        self.decoder_optimizer.zero_grad()
        hidden = (self.decoder.init_hidden(), self.decoder.init_cell())
        stack = self.decoder.init_stack()
        loss = 0
        for i in range(len(instr)):
            output, hidden, stack = self.decoder(instr[i], 
                                               hidden,
                                               stack)
            loss_s = self.criterion(output, tstr[i].unsqueeze(0))
            loss+= loss_s
            
        loss.backward()
        self.decoder_optimizer.step()
        return loss.item()/len(instr)
    
    def train(self, end, start=0):
        epochs = start+end
        for epoch in range(start, epochs):
            # sample a random batch
            idx = [x for x in range(len(self.smis))]
            random.shuffle(idx)
            idx = idx[:self.batch_size]
            dt_batch = [self.smis[x] for x in idx]
            dt_batch = self.Smi2Tensor(dt_batch)
            gene_batch = [torch.tensor(self.gene[x].astype(float), dtype=torch.float32) for x in idx]

            pbar = tqdm(range(self.batch_size), position=0, leave=True)
            pbar.set_description('Epoch %d'%epoch)
            loss_m = 0
            
            for i in pbar:
                smi = dt_batch[i]
                inp = smi[:-1].to(self.device)
                tar = smi[1:].to(self.device)
                gene = gene_batch[i].to(self.device)
                # no gene info
                if gene.sum() == 0:
                    loss = self.D_train_one_step(inp, tar)
                # gene info
                else:
                    loss = self.ED_train_one_step(inp, tar, gene)
                    
                loss_m += loss
            
            loss_m = loss_m/self.batch_size            
            self.logs.add_scalar('Loss', loss_m, epoch)

            if (epoch+1)%10 == 0:
                self.save_model(self.path, epoch)
                v_r_d, v_r_g = self.check_valid_ratio()
                self.logs.add_scalars('Valid_ratio', 
                                      {'D_sample': v_r_d,
                                       'G_sample': v_r_g}, epoch)
    
    def check_valid_ratio(self, n =100):
        pbar = tqdm(range(n), leave = True, position = 0)
        a = []
        for i in pbar:
            s = self.D_sample()
            s = Chem.MolFromSmiles(s[1:-1])
            if s != None:
                try:
                    s = Chem.MolToSmiles(s)
                    if s!= '':
                        a.append(s)
                except:
                    pass
        v_r_d = float(len(a))/n
        
        pbar = tqdm(range(n), leave = True, position = 0)
        b = []
        for i in pbar:
            s = self.G_sample()
            s = Chem.MolFromSmiles(s[1:-1])
            if s != None:
                try:
                    s = Chem.MolToSmiles(s)
                    if s!= '':
                        b.append(s)
                except:
                    pass
        v_r_g = float(len(b))/n
                
        return v_r_d, v_r_g

    
    def Smi2Tensor(self, smis):
        out = []
        long_token = {'Br':'A', 'Cl':'D'}
        if len(smis) > 100:
            pbar = tqdm(range(len(smis)), position=0, leave=True)
            pbar.set_description('Smi2Tensor')
        else:
            pbar = range(len(smis))
        for i in pbar:
            smi = smis[i]
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

    def D_sample(self, start = '<', end = '>', smi_len = 100):
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            hidden = (self.decoder.init_hidden(), self.decoder.init_cell())
            stack = self.decoder.init_stack()
        
            new_sample = start
            start = self.Smi2Tensor([start])[0].to(self.device)
        
            for i in range(len(start)-1):
                _, hidden, stack = self.decoder(start[i], hidden, stack)
            
            inpt = start[-1]
        
            for i in range(smi_len):
                out, hidden, stack = self.decoder(inpt, hidden, stack)
            
                p = torch.softmax(out, dim = 1)
                top_i = torch.multinomial(p.view(-1), 1)[0]
            
                new_char = self.rev_map[int(top_i)]
                new_sample+= new_char
            
                inpt = self.Smi2Tensor([new_char])[0].to(self.device)
                if new_char == end:
                    break
        return new_sample
        
    def G_sample(self, start = '<', end = '>', smi_len = 100, gene = None):
        
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            if type(gene) == type(None):
                gene = torch.tensor(self.gene[23333].astype(float), dtype=torch.float32).to(self.device)
            
            encoder_outputs = self.encoder(gene)
        
            decoder_hidden = (self.init_hidden(encoder_outputs.mean(dim=1)).view(1,1,-1), 
                              self.init_cell(encoder_outputs.mean(dim=1)).view(1,1,-1))
          
            stack = self.decoder.init_stack()

        
            new_sample = start
            start = self.Smi2Tensor([start])[0].to(self.device)
                
            inpt = start[-1]
        
            for i in range(smi_len):
                out, decoder_hidden, stack, attn_weight = self.decoder(inpt, 
                                                                       decoder_hidden, 
                                                                       stack,
                                                                       encoder_outputs.squeeze(0))
                    
                p = torch.softmax(out, dim = 1)
                top_i = torch.multinomial(p.view(-1), 1)[0]
            
                new_char = self.rev_map[int(top_i)]
                new_sample+= new_char
            
                inpt = self.Smi2Tensor([new_char])[0].to(self.device)
                if new_char == end:
                    break
            return new_sample
        
    def save_model(self, path, epoch):
        torch.save(self.encoder.state_dict(), path+'model.encoder.E%d.pkl'%(epoch+1))
        torch.save(self.decoder.state_dict(), path+'model.decoder.E%d.pkl'%(epoch+1))

            
    def load_model(self, path, epoch):
        self.encoder.load_state_dict(torch.load(path+'model.encoder.E%d.pkl'%(epoch+1), 
                                                map_location=self.device))
        self.decoder.load_state_dict(torch.load(path+'model.decoder.E%d.pkl'%(epoch+1), 
                                                map_location=self.device))

