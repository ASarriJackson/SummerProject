import os
import sys

# Get the directory of the current script
current_script_path = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of the current script to sys.path
parent_directory = os.path.dirname(current_script_path)
sys.path.append(parent_directory)

from helper_fun import *
import helper_fun
from Split_functions_classify.splits_two import *
from Split_functions_classify.splits_three import *
from Split_functions_classify.splits_ten import *

SEED = 22
seed_everything(SEED)

compounds = pd.read_csv('COVID_MOONSHOT/compounds_filtered.csv')
# compounds.head(2)

two_split(compounds)
three_split(compounds)
ten_split(compounds).head()
compound_df = compounds.copy()
compound_df["maccs"] = compound_df["SMILES"].apply(smiles_to_fp,)

# Specify models for 3 bin types
model_SVM_2 = svm.SVC(kernel="rbf", C=1, gamma=0.1, probability=True)
model_SVM_3 = svm.SVC(kernel="rbf", C=1, gamma=0.1, probability=True)
model_SVM_10 = svm.SVC(kernel="rbf", C=1, gamma=0.1, probability=True)

# Fit model on single split
# performance_measures = model_training_and_validation(model_SVM_2, "SVM", splits)
# Append SVM model
# models.append({"label": "Model_SVM", "model": model_SVM})
# # Plot roc curve
# plot_roc_curves_for_models(models, static_test_x, static_test_y);