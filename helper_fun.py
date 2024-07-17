#import necessary packages
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import matplotlib.patches as mpatches
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, PandasTools, rdFingerprintGenerator, AllChem
from rdkit.ML.Cluster import Butina
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from warnings import filterwarnings
import random

#Function to calc ro5 
def filter_ro5_properties(smiles):
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate Ro5-relevant chemical properties
    molecular_weight = Descriptors.ExactMolWt(molecule)
    n_hba = Descriptors.NumHAcceptors(molecule)
    n_hbd = Descriptors.NumHDonors(molecule)
    logp = Descriptors.MolLogP(molecule)
    # Check if Ro5 conditions fulfilled
    conditions = [molecular_weight <= 500, n_hba <= 10, n_hbd <= 5, logp <= 5]
    ro5_fulfilled = sum(conditions) == 4
    # Return True no conditions are violated
    
    return pd.Series(
        [ro5_fulfilled],
        index=["ro5_fulfilled"],
    )

#Copied from T022
def smiles_to_fp(smiles, method="morgan3", n_bits=2048):
    """
    Encode a molecule from a SMILES string into a fingerprint.

    Parameters
    ----------
    smiles : str
        The SMILES string defining the molecule.

    method : str
        The type of fingerprint to use. Default is MACCS keys.

    n_bits : int
        The length of the fingerprint.

    Returns
    -------
    array
        The fingerprint array.
    """

    # Convert smiles to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)

    if method == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if method == "morgan2":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.array(fpg.GetCountFingerprint(mol))
    if method == "morgan3":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
        return np.array(fpg.GetCountFingerprint(mol))
    else:
        print(f"Warning: Wrong method specified: {method}." " Default will be used instead.")
        return np.array(fpg.GetCountFingerprint(mol))
    

def convert_ic50_to_pic50(IC50_value):
    pIC50_value = 6 - math.log10(IC50_value)
    return pIC50_value

