from functools import partial
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle
import sklearn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

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

def Melanin(mol):
    """Calculates predicted melanin binding"""
    model = pickle.load(open('pickle_model_melanin.pkl', 'rb'))
    ds = np.array(get_descriptors([Chem.MolToSmiles (mol)])).reshape(1, -1)
    mel = model.predict_proba(ds)[0][1]

    return mel 


def Irritation(mol):
    model = pickle.load(open('pickle_model_irritation.pkl', 'rb'))
    ds = np.array(get_descriptors([Chem.MolToSmiles (mol)])).reshape(1, -1)
    irr = model.predict_proba(ds)[0][1]

    return irr



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
    #"""Calculates predicted corneal permeability""" 
    model = pickle.load(open('pickle_model_corneal.pkl', 'rb')) 
    ds = smi_to_descriptors([Chem.MolToSmiles(mol)]) 
    cor = model.predict(ds) 
    return cor[0]


