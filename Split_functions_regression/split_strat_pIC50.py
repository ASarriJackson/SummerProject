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
def strat_pIC50_split(data_x = compounds["morgan"], data_y = compounds["f_avg_pIC50"], test_size=0.2):
    bin_edges = np.arange(data_y.min(), data_y.max() + 0.5, 0.5)
    data_y = pd.cut(data_y, bins=bin_edges, include_lowest=True)

    # Split the data into train and test sets
    x_train_pIC, x_test_pIC, y_train_pIC, y_test_pIC = train_test_split(
    data_x, data_y, test_size=test_size,
    stratify=data_y)
    return train_x, test_x, train_y, test_y

def strat_pIC50_split_val(data_x= compounds["morgan"], data_y=compounds['f_avg_pIC50'], train_ratio=0.8, val_ratio=0.1, test_ratio=1-train_ratio-val_ratio, random_state=42):
    bin_edges = np.arange(data_y.min(), data_y.max() + 0.5, 0.5)
    data_y = pd.cut(data_y, bins=bin_edges, include_lowest=True)

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=1 - train_ratio, random_state=random_state, stratify=data_y)
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=test_ratio/(test_ratio + val_ratio), stratify=data_y) 
    return train_x, test_x, val_x, train_y, test_y, val_y