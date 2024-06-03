from functools import partial
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle
import sklearn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import torch
from kan.KAN import KAN

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


def get_descriptors(smiles): 
    descriptors = [] 
    mols = [Chem.MolFromSmiles(smile) for smile in smiles] 
    descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties()) 
    get_ds = rdMolDescriptors.Properties(descriptor_names) 
    for mol in mols: 
        ds = get_ds.ComputeProperties(mol)
        descriptors.append(ds)
        return descriptors

def Irritation(mol):
    model = pickle.load(open('pickle_model_irritation.pkl', 'rb'))
    ds = np.array(get_descriptors([Chem.MolToSmiles (mol)])).reshape(1, -1)
    irr = model.predict_proba(ds)[0][1]
    return irr

def get_fps(smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = []
    for i in range(len(mols)):
        fp = [int(bit) for bit in MACCSkeys.GenMACCSKeys(mols[i]).ToBitString()][:166]
        fps.append(fp)
    fps_array = np.array(fps, dtype=np.int32)  # Convert list of NumPy arrays to a single NumPy array
    fps_tensor = torch.tensor(fps_array)
    return fps_tensor


def Melanin(mol):
    model = KAN(width=[166,1,1], grid=10, k=3, seed=2000)
    model.load_state_dict(torch.load('kan_model.pth'))
    fps_tensor = get_fps([Chem.MolToSmiles(mol)])
    mel = model(fps_tensor)[:, 0].detach().numpy()
    return mel[0]

def smi_to_descriptors(smiles):
    descriptors = []
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    for mol in mols:
        lipinskiHBD = Descriptors.NumHDonors(mol)
        NumHBD = Descriptors.NumHDonors(mol)
        NumHeterocycles = Chem.rdMolDescriptors.CalcNumHeterocycles(mol)
        CrippenClogP = Descriptors.MolLogP(mol)
        hallKierAlpha = Descriptors.HallKierAlpha(mol)
        descriptor_values = [lipinskiHBD, NumHBD, NumHeterocycles, CrippenClogP, hallKierAlpha]
        descriptors.append(descriptor_values)
    return descriptors

def Corneal(mol): 
    model = pickle.load(open('pickle_model_corneal.pkl', 'rb')) 
    ds = smi_to_descriptors([Chem.MolToSmiles(mol)]) 
    cor = model.predict(ds) 
    return cor[0]


