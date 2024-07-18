
from helper_fun import *
# Import our filterted compounds
# # Define a function that adds an additional column defining the 2 bins
def two_split(compounds, target="f_avg_pIC50"):
    compounds["bin_2"] = pd.qcut(compounds[target],2,labels=[0,1])
    return compounds

#filt_comp_2_split = two_split(filt_comp)
