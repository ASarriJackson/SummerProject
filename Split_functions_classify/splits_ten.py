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

def ten_split(filt_comp, target = 'f_avg_pIC50'): #arg1: name of table, arg2: target (default = 'f_avg_pIC50')
    filt_comp['bin_10'] = pd.qcut(filt_comp[target], 13, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  duplicates='drop', precision=100)
    return filt_comp

#plot a histogram of the distribution of the bin values afterpassing filt_comp through three_split
# ten_split(filt_comp).head()
# bin_counts = ten_split(filt_comp)['10_bin'].value_counts()
# print(bin_counts)   