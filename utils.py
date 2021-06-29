import pandas as pd
import numpy as np
import pickle
import random
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import trange, tqdm
sns.set_style('white')
import warnings
warnings.filterwarnings('once')
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'

from scipy.special import softmax
from sklearn import metrics

from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, RWMol, CombineMols
from rdkit.Chem import Recap
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True


class MolSplit():
    def __init__(self, smis):
        super(MolSplit, self).__init__()
        self.smis = smis
        self.mols = []
        pbar = tqdm(range(len(self.smis)), position=0, leave=True)
        pbar.set_description('Converting Smis to Mols')
        for i in pbar:
            self.mols.append(Chem.MolFromSmiles(self.smis[i]))
            
        self.scaffolds = []
        self.linker = []
        self.sidechain = []

    def decomp(self):
        pbar = tqdm(range(len(self.mols)), position=0, leave=True)
        pbar.set_description('Decompose Mols')
        for i in pbar:
            mol = self.mols[i]
            de = Chem.Recap.RecapDecompose(mol)
            frag = [x.mol for x in de.GetLeaves().values()]
            for i in frag:
                if i.GetRingInfo().NumRings() > 0:
                    self.scaffolds.append(Chem.MolToSmiles(i))
                else:
                    s = Chem.MolToSmiles(i)
                    star_count = len(s.split('*'))-1
                    if star_count == 1:
                        self.sidechain.append(s)
                    else:
                        self.linker.append(s)
        self.sanitize()
        
    def sanitize(self):
        self.scaffolds = list(set(self.scaffolds))
        self.linker = list(set(self.linker))
        self.sidechain = list(set(self.sidechain))
        
    def randomize(self, num = 20):
        pbar = tqdm(range(len(self.scaffolds)), position=0, leave=True)
        pbar.set_description('Generating random scaffolds')
        random_scaffolds = []
        for i in pbar:
            mol = Chem.MolFromSmiles(self.scaffolds[i])
            rds = list(set([Chem.MolToSmiles(mol, canonical=False, doRandom=True) for _ in range(num)]))
            random_scaffolds+=rds
        random.shuffle(random_scaffolds)
        self.random_scaffolds = random_scaffolds
        


def MolJoin(scaffold, decoration):
        
    def __join(scaffold, decoration):
        scaffold = Chem.MolFromSmiles(scaffold)
        decoration = Chem.MolFromSmiles(decoration)
        combined = RWMol(CombineMols(scaffold,decoration))
        s_at_num = len([at for at in scaffold.GetAtoms() 
                    if at.GetSymbol() == '*'])
        d_at_num = len([at for at in decoration.GetAtoms() 
                    if at.GetSymbol()])
        attach_comb = [at for at in combined.GetAtoms() 
                   if at.GetSymbol() == '*']
        attachment = [attach_comb[:s_at_num][0], 
                  attach_comb[s_at_num:][0]]
        neighbors = [at.GetNeighbors()[0] for at in attachment]
        bonds = [at.GetBonds()[0] for at in attachment]
        bond_type = Chem.BondType.SINGLE
        if any(bond for bond in bonds if bond.GetBondType() == Chem.BondType.DOUBLE):
            bond_type = Chem.BondType.DOUBLE
        combined.AddBond(neighbors[0].GetIdx(), neighbors[1].GetIdx(), bond_type)
        combined.RemoveAtom(attachment[0].GetIdx())
        combined.RemoveAtom(attachment[1].GetIdx())
        out = Chem.MolToSmiles(combined)
        return out
    
    scaffold = scaffold[1:-1]
    decoration = [x for x in decoration[1:-1].split('|')]
    decoration = sorted(decoration, key=lambda x: x.count('*'), 
                        reverse=True)
    for deco in decoration:
        try:
            scaffold = __join(scaffold, deco)
        except:
            #print(scaffold, deco)
            return ''
        
    return scaffold