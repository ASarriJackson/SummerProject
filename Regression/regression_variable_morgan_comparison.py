from pathlib import Path
import sys
# Get the directory of the current file
current_file_path = Path(__file__)

# Get the parent directory of the current directory (i.e., SummerProject)
parent_directory = current_file_path.parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_directory))
from helper_fun import *

# Construct the path to the compounds_filtered.csv file
compounds_file_path = parent_directory / 'COVID_MOONSHOT' / 'compounds_filtered.csv'
compounds = pd.read_csv(compounds_file_path)
# Silence some expected warnings
filterwarnings("ignore")

from Split_functions_regression.split_furthest_cluster import *
from Split_functions_regression.split_hierarchical_cluster import *
from Split_functions_regression.split_random import *
from Split_functions_regression.split_strat_pIC50 import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import seaborn as sns

# Neural network specific libraries
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint


SEED = 42
seed_everything(SEED)

compounds["Fingerprints"] = compounds["SMILES"].apply(smiles_to_fp)
compounds["morgan2"] = compounds["SMILES"].apply(smiles_to_fp, method="morgan2", n_bits=2048)
x_train_rand, x_test_rand, y_train_rand, y_test_rand = random_split(compounds["Fingerprints"], compounds["f_avg_pIC50"])
x_train_strat, x_test_strat, y_train_strat, y_test_strat = strat_pIC50_split(compounds)
x_train_hi, x_test_hi, y_train_hi, y_test_hi = split_hierarchical_clusters(compounds, test_size=0.2, random_state=42)
x_train_noise, x_test_noise, y_train_noise, y_test_noise = UMAP_noise_split(compounds)
x_train_fur, x_test_fur, y_train_fur, y_test_fur = furthest_cluster_split(compounds)

x_train_rand_m2, x_test_rand_m2, y_train_rand_m2, y_test_rand_m2 = random_split(compounds["morgan2"], compounds["f_avg_pIC50"])
x_train_strat_m2, x_test_strat_m2, y_train_strat_m2, y_test_strat_m2 = strat_pIC50_split(compounds,fingerprint_column="morgan2")
x_train_hi_m2, x_test_hi_m2, y_train_hi_m2, y_test_hi_m2 = split_hierarchical_clusters(compounds,fingerprint_method="morgan2", test_size=0.2, random_state=42)
x_train_noise_m2, x_test_noise_m2, y_train_noise_m2, y_test_noise_m2 = UMAP_noise_split(compounds,fingerprint_column="morgan2")
x_train_fur_m2, x_test_fur_m2, y_train_fur_m2, y_test_fur_m2 = furthest_cluster_split(compounds,fingerprint_column="morgan2")