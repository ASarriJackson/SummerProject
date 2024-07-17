#!/usr/bin/env python
# coding: utf-8

# ### Copied from PDF

# In[114]:


# jupyter nbconvert --to script helper_fun.ipynb


# In[115]:


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

from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import random

from scipy.stats import spearmanr
from scipy.cluster import hierarchy

# Neural network specific libraries
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

get_ipython().run_line_magic('matplotlib', 'inline')

# Silence some expected warnings
filterwarnings("ignore")


from teachopencadd.utils import seed_everything
# Fix seed for reproducible results
SEED = 22
seed_everything(SEED)


# In[116]:


#Helper function to compute descriptors for a single molecule

def compute_descriptors(molecule):
    descriptors = {d[0]: d[1](molecule) for d in Descriptors.descList}
    descriptors = pd.Series(descriptors)
    return descriptors


# ## Function adapted from T002 calculate_ro5_properties

# #### Returns only a pd series containing boolean value for r05_fulfilled, results are then added as a column to compounds table

# In[117]:


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


# In[118]:


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


# In[119]:


def tanimoto_distance_matrix(fp_list):
    """Calculate distance matrix for fingerprint list"""
    dissimilarity_matrix = []
    # Notice how we are deliberately skipping the first and last items in the list
    # because we don't need to compare them against themselves
    for i in range(1, len(fp_list)):
        # Compare the current fingerprint against all the previous ones in the list
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix


# In[120]:


def cluster_fingerprints(fingerprints, cutoff=0.25):

    # Calculate Tanimoto distance matrix
    distance_matrix = tanimoto_distance_matrix(fingerprints)
    # Now cluster the data with the implemented Butina algorithm:
    clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


# In[121]:


def intra_tanimoto(fps_clusters):
    """Function to compute Tanimoto similarity for all pairs of fingerprints in each cluster"""
    intra_similarity = []
    # Calculate intra similarity per cluster
    for cluster in fps_clusters:
        # Tanimoto distance matrix function converted to similarity matrix (1-distance)
        intra_similarity.append([1 - x for x in tanimoto_distance_matrix(cluster)])
    return intra_similarity


# In[122]:


# Define function that transforms SMILES strings into ECFPs
def ECFP_from_smiles2(smiles, R=2, L=2**10, use_features=False, use_chirality=False):
    """
    Inputs:
    - smiles ... SMILES string of input compound
    - R ... maximum radius of circular substructures
    - L ... fingerprint-length
    - use_features ... if false then use standard DAYLIGHT atom features, if true then use pharmacophoric atom features
    - use_chirality ... if true then append tetrahedral chirality flags to atom features
    
    Outputs:
    - np.array(feature_list) ... ECFP with length L and maximum radius R
    """
    molecule = Chem.MolFromSmiles(smiles)

    # Create a Morgan fingerprint generator with the specified parameters
    morgan_gen = GetMorganGenerator(radius=R, fpSize=L, includeChirality=use_chirality)
    feature_list = morgan_gen.GetFingerprint(molecule)
        
    return np.array(feature_list)


# In[123]:


def convert_ic50_to_pic50(IC50_value):
    pIC50_value = 6 - math.log10(IC50_value)
    return pIC50_value


# In[124]:


#Copied verbatim from T022
def neural_network_model(hidden1, hidden2):
    """
    Creating a neural network from two hidden layers
    using ReLU as activation function in the two hidden layers
    and a linear activation in the output layer.

    Parameters
    ----------
    hidden1 : int
        Number of neurons in first hidden layer.

    hidden2: int
        Number of neurons in second hidden layer.

    Returns
    -------
    model
        Fully connected neural network model with two hidden layers.
    """

    model = Sequential()
    # First hidden layer
    model.add(Dense(hidden1, activation="relu", name="layer1"))
    # Second hidden layer
    model.add(Dense(hidden2, activation="relu", name="layer2"))
    # Output layer
    model.add(Dense(1, activation="linear", name="layer3"))

    # Compile model
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse", "mae"])
    return model


# # Step 1: Data Collection

# In[125]:


assays = pd.read_csv("covid_submissions_all_info.csv", header = 0, usecols=["CID", "f_avg_IC50", "r_avg_IC50"])
assays.drop_duplicates("CID", keep="first", inplace=True)
# print(f"Initial Number of Molecules: {len(assays)}")
assays.head()

# missing_values_count = assays.isnull().sum()
# print(missing_values_count)

# full_assays = pd.read_csv("covid_submissions_all_info.csv")
# print(f"Number of rows in full CSV: {len(full_assays)}")
#This still gives 20997???


# In[126]:


#filter out NaN for r and f avgIC50 columns
assays.dropna(axis=0, how="any", subset = ["r_avg_IC50", "f_avg_IC50"], inplace=True)
print(f"Initial Number of Molecules: {len(assays)}")
# assays.head()
# print(type(assays_df))
#looks good


# In[127]:


#filter out compounds w NaN value in f_avg and r_avg
#drop duplicates
compounds = pd.read_csv("covid_submissions_all_info.csv", usecols=["CID", "SMILES", "r_avg_IC50", "f_avg_IC50"])
compounds.dropna(axis=0, how="any", subset = ["r_avg_IC50", "f_avg_IC50"], inplace=True)
compounds.drop(["r_avg_IC50", "f_avg_IC50"], axis=1)
compounds.drop_duplicates("CID", keep="first", inplace=True)
# print(f"Initial Number of Molecules: {len(compounds)}")
# compounds.head()


# In[128]:


merge_ca = pd.merge(assays, compounds, on="CID", how ="left")
# print(len(merge_ca))
merge_ca.head()


# In[129]:


#QUERY CHECK 1
#SMILES from the compounds table

CID_Query = "DAR-DIA-23aa0b97-19"
assays_query = assays.loc[assays["CID"]=="DAR-DIA-23aa0b97-19"]
assays_query


# In[130]:


#QUERY CHECK 2
#affinity data from the assays table
compounds_query = compounds.loc[compounds["CID"] == "DAR-DIA-23aa0b97-19"]
compounds_query


# # Step 2: Data Exploration
# ### Add descriptors to compound table (only)

# In[131]:


#This code is taken from T002
# RDKit to compute molecular descriptors, add each property to a new column in comp table

PandasTools.AddMoleculeColumnToFrame(compounds, "SMILES")
compounds["molecular_weight"] = compounds["ROMol"].apply(Descriptors.ExactMolWt)
compounds["n_hba"] = compounds["ROMol"].apply(Descriptors.NumHAcceptors)
compounds["n_hbd"] = compounds["ROMol"].apply(Descriptors.NumHDonors)
compounds["logp"] = compounds["ROMol"].apply(Descriptors.MolLogP)
# compounds.head()


# In[132]:


# Filtering with R05
compounds["Ro5_fulfilled"] = compounds["SMILES"].apply(filter_ro5_properties)
compounds = compounds[compounds["Ro5_fulfilled"] == True]
print(f"Number of molecules that satisfy Ro5: {len(compounds)}")
# compounds.head()


# ### Fingerprints

# In[133]:


#Must create an array of compounds to produce fingerprints
comp_arr = []
for _, CID, smiles in compounds[["CID", "SMILES"]].itertuples():
    comp_arr.append((Chem.MolFromSmiles(smiles), CID))
comp_arr[:3]


# In[134]:


fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
compounds["morgan"] = compounds["ROMol"].map(fpg.GetFingerprint)
fingerprints = [rdkit_gen.GetFingerprint(mol) for mol, idx in comp_arr]
# print(f"Number of fingerprints: {len(fingerprints)}")
compounds.head()
#how can we make clusters from this information?
#does morgan need to be a column or could make a list ?? easier to pass into a function


# ### Clusters For Morgan Fingerprints
# ##### Can optimize clusters for model training by changing cut-off 
# 

# In[135]:


# for cutoff in np.arange(0.2, 0.31, 0.025):
#     clusters = cluster_fingerprints(fingerprints, cutoff=cutoff)
#     fig, ax = plt.subplots(figsize=(15, 4))
#     ax.set_title(f"Threshold: {cutoff:3.3f}")
#     ax.set_xlabel("Cluster index")
#     ax.set_ylabel("Number of molecules")
#     ax.bar(range(1, len(clusters) + 1), [len(c) for c in clusters], lw=5)
#     display(fig)# Run the clustering procedure for the dataset


# ##### Here I changed cut off by 0.05 each plot, we are ideally looking for a plot with few singletons (few low bars) and no extreme distribution of cluster sizes (no huge steps). We are looking for a smooth distribution.
# 
# ##### More analysis is required

# In[136]:


# # Run the clustering procedure for the dataset
# for i in np.arange(0.2, 0.31, 0.025):
#     clusters = cluster_fingerprints(fingerprints, cutoff=i)

#     # Give a short report about the numbers of clusters and their sizes
#     num_clust_g1 = sum(1 for c in clusters if len(c) == 1)
#     num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
#     num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
#     num_clust_g50 = sum(1 for c in clusters if len(c) > 50)
#     num_clust_g100 = sum(1 for c in clusters if len(c) > 100)
#     print(f"\nCutoff: {i:3.3f}")
#     print("total # Molecules: ", len(fingerprints))
#     print("total # clusters: ", len(clusters))
#     print("# clusters with only 1 compound: ", num_clust_g1)
#     print("# clusters with >5 compounds: ", num_clust_g5)
#     print("# clusters with >25 compounds: ", num_clust_g25)
#     print("# clusters with >50 compounds: ", num_clust_g50)    
#     print("# clusters with >100 compounds: ", num_clust_g100)
# # NBVAL_CHECK_OUTPUT


# ##### I shall proceed with cutoff = 0.225 
# ##### This produces a relatively smooth distribution. 
# ##### Analysing clusters, we can first look at 5 molecules in the biggest cluster

# In[137]:


clusters = cluster_fingerprints(fingerprints, cutoff=0.225)


# In[138]:


#comp_arr is an array of tuples. First element in tuple is SMILES, second is CID
print("Five molecules from largest cluster:")
print(f"Size of Cluster 0: {clusters[0]}")
# Draw molecules
Draw.MolsToGridImage(
    [comp_arr[i][0] for i in clusters[0][:5]],
    legends=[comp_arr[i][1] for i in clusters[0][:5]],
    molsPerRow=5,
)


# In[139]:


# print("Five molecules from second largest cluster:")
# print(f"Size of Cluster 1: {clusters[1]}")
# # Draw molecules
# Draw.MolsToGridImage(
#     [comp_arr[i][0] for i in clusters[1][:5]],
#     legends=[comp_arr[i][1] for i in clusters[1][:5]],
#     molsPerRow=5,
# )


# In[140]:


# print("One molecule from first 5 clusters:")
# # Draw molecules
# Draw.MolsToGridImage(
#     [comp_arr[clusters[i][0]][0] for i in range(5)],
#     legends=[comp_arr[clusters[i][0]][1] for i in range(5)],
#     molsPerRow=5,
# )


# ### Intra-cluster Tanimoto similarities

# In[141]:


# code from T005

# Recompute fingerprints for 10 first clusters
#cluster 0 is the biggest, clusters decrease in size and index increases
mol_fps_per_cluster = []
for cluster in clusters[:10]:
    mol_fps_per_cluster.append([rdkit_gen.GetFingerprint(comp_arr[i][0]) for i in cluster])

# Compute intra-cluster similarity
intra_sim = intra_tanimoto(mol_fps_per_cluster)


# In[142]:


# Violin plot with intra-cluster similarity

# fig, ax = plt.subplots(figsize=(10, 5))
# indices = list(range(10))
# ax.set_xlabel("Cluster index")
# ax.set_ylabel("Similarity")
# ax.set_xticks(indices)
# ax.set_xticklabels(indices)
# ax.set_yticks(np.arange(0.6, 1.0, 0.1))
# ax.set_title("Intra-cluster Tanimoto similarity", fontsize=13)
# r = ax.violinplot(intra_sim, indices, showmeans=True, showmedians=True, showextrema=False)
# r["cmeans"].set_color("red")
# # mean=red, median=blue


# ### Interpretation
# ##### The red line represents the mean and the blue line represents the median. A longer 'violin' suggests that molecules in the cluster have a greater range in similarity values (aka large range in Tanimoto distance). Cluster 0 ranges in similarity from approx 0 to 1.
# 
# ##### A violin ranging over a low similarity score might suggest that the cut-off should decrease so that molecules within clusters become more similar criteria for a cluster is more particular. 
# 
# ##### Something weird is going on w cluster[6], what is this?

# ### Binding Affinity (pIC50 Values)

# In[143]:


#create new columns for f_avg_pIC50 and r_avg_pIC50
compounds["f_avg_pIC50"] = compounds.apply(lambda x: convert_ic50_to_pic50(x.f_avg_IC50), axis=1)
compounds["r_avg_pIC50"] = compounds.apply(lambda x: convert_ic50_to_pic50(x.r_avg_IC50), axis=1)
compounds.head()


# ##### Do we need to consider whether the compound is active or inactive? active compounds is only 56!

# In[144]:


# #Classify each compound as active or inactive (common pIC50 cutoff = 6.3)
# # Add column for activity
# compounds["active"] = np.zeros(len(compounds))

# # Mark every molecule as active with an pIC50 of >= 6.3, 0 otherwise
# compounds.loc[compounds[compounds.f_avg_pIC50 >= 6.3].index, "active"] = 1.0

# # NBVAL_CHECK_OUTPUT
# print("Number of active compounds:", int(compounds.active.sum()))
# print("Number of inactive compounds:", len(compounds) - int(compounds.active.sum()))


# In[145]:


# fig, axes = plt.subplots(1, 2, figsize=(14,8))
# compounds.hist(column="r_avg_pIC50", ax=axes[0])
# compounds.hist(column="f_avg_pIC50", ax=axes[1])


# ##### In both the r and f pIC50 tables we see a range of 4.0-7.0. Garrett mentioned above that values should range beween 4.0-6.0. 
# 
# ##### From the graphs, considering the range we are looking for, pIC50 = 4.0 seems to be greatly over-represented. This may lead to problems if the majority of training data has pIC50 = 4.0. 
# 
# ##### May cause bias during training. pIC50 must therefore be monitored within training and test data. 

# In[146]:


#export compounds to csv titled "filtered_compounds.csv"
compounds.to_csv("filtered_compounds.csv", index=False)
#save this csv to file Project0

