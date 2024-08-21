from functools import partial
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle
import sklearn
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
import torch
from kan.KAN import KAN
import xgboost as xgb

class Reward:
    def __init__(self, property, reward, weight=1.0, preprocess=None):
        self.property = property
        self.reward = reward
        self.weight = weight
        self.preprocess = preprocess

    def __call__(self, input):
        if self.preprocess:
            input = self.preprocess(input)
        property = self.property(input)
        reward = self.weight * self.reward(property)
        return reward, property


def identity(x):
    return x


def ReLU(x):
    return max(x, 0)


def HSF(x):
    return float(x > 0)


class OutOfRange:
    def __init__(self, lower=None, upper=None, hard=True):
        self.lower = lower
        self.upper = upper
        self.func = HSF if hard else ReLU

    def __call__(self, x):
        y, u, l, f = 0, self.upper, self.lower, self.func
        if u is not None:
            y += f(x - u)
        if l is not None:
            y += f(l - x)
        return y


class PatternFilter:
    def __init__(self, patterns):
        self.structures = list(filter(None, map(Chem.MolFromSmarts, patterns)))

    def __call__(self, molecule):
        return int(any(molecule.HasSubstructMatch(struct) for struct in self.structures))


def MolLogP(m):
    return rdMolDescriptors.CalcCrippenDescriptors(m)[0]


    
def get_fps(smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = []
    for i in range(len(mols)):
        fp = [int(bit) for bit in MACCSkeys.GenMACCSKeys(mols[i]).ToBitString()][:166]
        fps.append(fp)
    fps_array = np.array(fps, dtype=np.int32)  
    fps_tensor = torch.tensor(fps_array)
    return fps_tensor


def smi_to_maccs(smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = []
    for i in range(len(mols)):
        fp = [int(bit) for bit in MACCSkeys.GenMACCSKeys(mols[i]).ToBitString()][:167]
        fps.append(fp)
    fps_array = np.array(fps, dtype=np.int32).reshape(1, -1) 
    return fps_array



def get_maccs(smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = []
    for i in range(len(mols)):
        fp = [int(bit) for bit in MACCSkeys.GenMACCSKeys(mols[i]).ToBitString()][:167]
        fps.append(fp)
    fps_array = np.array(fps, dtype=np.int32)
    indices_to_drop = [11, 33, 47, 49, 50, 51, 55, 56, 58, 59, 60, 61, 63, 64, 67, 69, 70, 71, 73, 76, 80, 81, 88, 94, 102, 105, 106, 107, 110, 119, 124, 130, 134, 135, 143, 147]  
    new_array = np.delete(fps_array, indices_to_drop)
    maccs = new_array.reshape(1, -1)
    print(maccs.shape)
    return maccs 


def Melanin(mol):

    model = KAN(width=[166,1,1], grid=10, k=3, seed=2000)
    model.load_state_dict(torch.load('kan_model.pth'))
    fps_tensor = get_fps([Chem.MolToSmiles(mol)])
    mel = model(fps_tensor)[:, 0].detach().numpy()


    return mel[0]



def Irritation(mol):
    model = pickle.load(open('irritation.pkl', 'rb'))
    fps = smi_to_maccs([Chem.MolToSmiles(mol)])
    irr = model.predict_proba(fps)[0][1]

    return irr



def Corneal(mol): 
    #"""Calculates predicted corneal permeability""" 
    model = xgb.XGBRegressor(random_state=10)
    model.load_model('corneal.json') 
    maccs = get_maccs([Chem.MolToSmiles(mol)]) 
    cor = model.predict(maccs) 
    return cor[0]
