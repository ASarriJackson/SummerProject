# def three_splits()
from helper_fun import *
#read_csv file titled compounds_filtered.csv from directory COVID_MOONSHOT

def three_split(filt_comp, target = 'f_avg_pIC50'): #arg1: name of table, arg2: target (default = 'f_avg_pIC50')
    filt_comp['3_bin'] = pd.qcut(filt_comp[target], 3, labels = ['low', 'medium', 'high'])
    return filt_comp

#plot a histogram of the distribution of the bin values afterpassing filt_comp through three_split
# plt.hist(three_split(filt_comp)['3_bin'])