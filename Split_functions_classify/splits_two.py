

from helper_fun import *
# Import our filterted compounds
# # Define a function that adds an additional column defining the 2 bins
def two_split(compounds, target="f_avg_pIC50"):
    compounds["bin_2"] = pd.qcut(compounds[target],2,labels=[0,1])
    return compounds





from pathlib import Path
import sys
# Get the directory of the current file
current_file_path = Path(__file__)

# Get the parent directory of the current directory (i.e., SummerProject)
parent_directory = current_file_path.parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_directory))


#filt_comp_2_split = two_split(filt_comp)
