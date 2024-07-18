# def three_splits()
from helper_fun import *
#read_csv file titled compounds_filtered.csv from directory COVID_MOONSHOT
filt_comp = pd.read_csv("COVID_MOONSHOT/compounds_filtered.csv")

def ten_split(filt_comp, target = 'f_avg_pIC50'): #arg1: name of table, arg2: target (default = 'f_avg_pIC50')
    filt_comp['10_bin'] = pd.qcut(filt_comp[target], 13, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  duplicates='drop', precision=100)
    return filt_comp

#plot a histogram of the distribution of the bin values afterpassing filt_comp through three_split
# ten_split(filt_comp).head()
# bin_counts = ten_split(filt_comp)['10_bin'].value_counts()
# print(bin_counts)   