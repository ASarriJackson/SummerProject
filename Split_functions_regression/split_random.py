from pathlib import Path
import sys
# Get the directory of the current file
current_file_path = Path(__file__)

# Get the parent directory of the current directory (i.e., SummerProject)
parent_directory = current_file_path.parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_directory))
from helper_fun import *
#read_csv file titled compounds_filtered.csv from directory COVID_MOONSHOT

def random_split(data_x, data_y, test_size=0.2, random_state=42):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_size, random_state=random_state)
    return train_x, test_x, train_y, test_y