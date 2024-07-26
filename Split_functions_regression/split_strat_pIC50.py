from pathlib import Path
import sys
# Get the directory of the current file
current_file_path = Path(__file__)

# Get the parent directory of the current directory (i.e., SummerProject)
parent_directory = current_file_path.parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_directory))
from helper_fun import *

# Split data into train and test setsusing heirarchical clustering
def strat_pIC50_split(table, test_size=0.2, fingerprint_column = "Fingerprints", pIC50_column = "f_avg_pIC50"):
    data_x= table[fingerprint_column] 
    data_y = table[pIC50_column]
    bin_edges = np.arange(data_y.min(), data_y.max() + 0.5, 0.5)
    table["pIC50_range"] = pd.cut(data_y, bins=bin_edges, include_lowest=True)

    # Split the data into train and test sets
    x_train_pIC, x_test_pIC, y_train_pIC, y_test_pIC = train_test_split(
    data_x, data_y, test_size=test_size,
    stratify=table["pIC50_range"])
    return x_train_pIC, x_test_pIC, y_train_pIC, y_test_pIC

def strat_pIC50_split_val(table, train_ratio=0.8, val_ratio=0.1, random_state=42, fingerprint_column = "Fingerprints", pIC50_column = "f_avg_pIC50"):
    data_x= table[fingerprint_column]
    data_y = table[pIC50_column]
    bin_edges = np.arange(data_y.min(), data_y.max() + 0.5, 0.5)
    table["pIC50_range"] = pd.cut(data_y, bins=bin_edges, include_lowest=True)
    test_ratio = 1 - train_ratio - val_ratio,

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=1 - train_ratio, random_state=random_state, stratify=table["pIC50_range"])
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=test_ratio/(test_ratio + val_ratio), stratify=table["pIC50_range"]) 
    return train_x, test_x, val_x, train_y, test_y, val_y