#import necessary packages
from pathlib import Path
import math
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

from scipy.stats import spearmanr
import umap
from tqdm import tqdm
import hdbscan
import plotly.express as px


# import matplotlib.patches as mpatches
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, PandasTools, rdFingerprintGenerator, AllChem
from rdkit.ML.Cluster import Butina

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from sklearn import svm, metrics, clone
import sklearn.datasets
import sklearn.cluster as cluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import auc, accuracy_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.metrics import matthews_corrcoef, f1_score,classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


from itertools import cycle
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
def smiles_to_fp(smiles, method="maccs", n_bits=2048):
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


def seed_everything(seed=42):
    """Set the RNG seed in Python and Numpy"""
    import random
    import os
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
 
def model_performance(ml_model, test_x, test_y, verbose=True):
    """
    Helper function to calculate model performance

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    test_x: list
        Molecular fingerprints for test set.
    test_y: list
        Associated activity labels for test set.
    verbose: bool
        Print performance measure (default = True)

    Returns
    -------
    tuple:
        Accuracy, sensitivity, specificity, auc on test set.
    """

    # Prediction probability on test set
    test_prob = ml_model.predict_proba(test_x)[:, 1]

    # Prediction class on test set
    test_pred = ml_model.predict(test_x)

    # Performance of model on test set
    accuracy = accuracy_score(test_y, test_pred)
    sens = recall_score(test_y, test_pred)
    spec = recall_score(test_y, test_pred, pos_label=0)
    auc = roc_auc_score(test_y, test_prob)

    if verbose:
        # Print performance results
        # NBVAL_CHECK_OUTPUT        print(f"Accuracy: {accuracy:.2}")
        print(f"Sensitivity: {sens:.2f}")
        print(f"Specificity: {spec:.2f}")
        print(f"AUC: {auc:.2f}")

    return accuracy, sens, spec, auc

def model_training_and_validation(ml_model, name, splits, verbose=True):
    """
    Fit a machine learning model on a random train-test split of the data
    and return the performance measures.

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    name: str
        Name of machine learning algorithm: RF, SVM, ANN
    splits: list
        List of desciptor and label data: train_x, test_x, train_y, test_y.
    verbose: bool
        Print performance info (default = True)

    Returns
    -------
    tuple:
        Accuracy, sensitivity, specificity, auc on test set.

    """
    train_x, test_x, train_y, test_y = splits

    # Fit the model
    ml_model.fit(train_x, train_y)

    # Calculate model performance results
    accuracy, sens, spec, auc = model_performance(ml_model, test_x, test_y, verbose)

    return accuracy, sens, spec, auc


def plot_roc_curves_for_models(models, test_x, test_y, save_png=False):
    """
    Helper function to plot customized roc curve.

    Parameters
    ----------
    models: dict
        Dictionary of pretrained machine learning models.
    test_x: list
        Molecular fingerprints for test set.
    test_y: list
        Associated activity labels for test set.
    save_png: bool
        Save image to disk (default = False)

    Returns
    -------
    fig:
        Figure.
    """

    fig, ax = plt.subplots()

    # Below for loop iterates through your models list
    for model in models:
        # Select the model
        ml_model = model["model"]
        # Prediction probability on test set
        test_prob = ml_model.predict_proba(test_x)[:, 1]
        # Prediction class on test set
        test_pred = ml_model.predict(test_x)
        # Compute False postive rate and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(test_y, test_prob)
        # Calculate Area under the curve to display on the plot
        auc = roc_auc_score(test_y, test_prob)
        # Plot the computed values
        ax.plot(fpr, tpr, label=(f"{model['label']} AUC area = {auc:.2f}"))

    # Custom settings for the plot
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    # Save plot
    if save_png:
        fig.savefig(f"{DATA}/roc_auc", dpi=300, bbox_inches="tight", transparent=True)
    return fig


def plot_roc_for_multi_class(model,static_test_x,static_train_y,static_test_y,bins_label,one_vs_rest=True,micro=True):
    test_prob = model.predict_proba(static_test_x)
    
    label_binarizer = LabelBinarizer().fit(static_train_y)
    y_onehot_test = label_binarizer.transform(static_test_y)
    
    fpr, tpr, roc_auc = dict(), dict(), dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(),test_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"],tpr["micro"])
    
    fig, ax = plt.subplots(figsize=(10,10))
    if micro == True:
        plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
        color = "deeppink",
        linestyle=":",
        linewidth=4,
    )
    if one_vs_rest == True:
        colors = cycle(["blue","green","orange","red","violet","brown","aqua","black","darkblue","purple"])
        for class_id, color in zip(range(len(bins_label)), colors):
            RocCurveDisplay.from_predictions(
                y_onehot_test[:, class_id],
                test_prob[:, class_id],
                name=f"ROC curve for {bins_label[class_id]}",
                color=color,
                ax=ax,
                plot_chance_level=(class_id == len(bins_label)-1),
            )
    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC curves for multiclass"
    )
    if one_vs_rest==False:
        plt.plot([0, 0], [0, 1.0], color="black", label="Chance level (AUC = 0.5)", linestyle="--")
        plt.legend(
            loc = "lower right"
        )
    return None


def calculate_micro_auc(model,static_test_x,static_train_y,static_test_y):
    test_prob = model.predict_proba(static_test_x)
    
    label_binarizer = LabelBinarizer().fit(static_train_y)
    y_onehot_test = label_binarizer.transform(static_test_y)
    
    fpr, tpr, roc_auc = dict(), dict(), dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(),test_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"],tpr["micro"])
    
    return auc(fpr["micro"],tpr["micro"])
