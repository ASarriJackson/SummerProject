from pathlib import Path
import sys
# Get the directory of the current file
current_file_path = Path(__file__)

# Get the parent directory of the current directory (i.e., SummerProject)
parent_directory = current_file_path.parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_directory))
from helper_fun import *

def three_split(compounds, target = 'f_avg_pIC50'): #arg1: name of table, arg2: target (default = 'f_avg_pIC50')
    compounds['three_bin'] = pd.qcut(compounds[target], 3, labels = ['low', 'medium', 'high'])
    return compounds

#plot a histogram of the distribution of the bin values afterpassing filt_comp through three_split
# plt.hist(three_split(filt_comp)['3_bin'])