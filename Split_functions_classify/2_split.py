from helper_fun import *
# Import our filterted compounds
filt_comp = pd.read_csv("COVID_MOONSHOT/compounds_filtered.csv")
# Define a function that adds an additional column defining the 2 bins
def two_split(compounds, target="f_avg_pIC50"):
    compounds["2_bin"] = pd.qcut(compounds[target],2,labels=[0,1])
    return compounds

#filt_comp_2_split = two_split(filt_comp)
