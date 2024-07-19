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
ten_split(compounds)

compound_df = compounds.copy()
compound_df["maccs"] = compound_df["SMILES"].apply(smiles_to_fp,)

# Bins 2, 3, 10

#change variables to _2 at end
fingerprint_to_model_2 = compound_df.maccs.tolist()
label_to_model_2 = compound_df.bin_2.tolist()

# Split data randomly in train and test set
# note that we use test/train_x for the respective fingerprint splits
# and test/train_y for the respective label splits
(
    static_train_x_2,
    static_test_x_2,
    static_train_y_2,
    static_test_y_2,
) = train_test_split(fingerprint_to_model_2, label_to_model_2, test_size=0.2, random_state=SEED)
splits_2 = [static_train_x_2, static_test_x_2, static_train_y_2, static_test_y_2]
# NBVAL_CHECK_OUTPUT
print("Training data size:", len(static_train_x_2))
print("Test data size:", len(static_test_x_2))


# Set model parameter for random forest
param_2 = {
    "n_estimators": 100,  # number of trees to grows
    "criterion": "entropy",  # cost function to be optimized for a split
}
model_RF_2 = RandomForestClassifier(**param_2)

#change variables to _3 at end
fingerprint_to_model_3 = compound_df.maccs.tolist()
label_to_model_3 = compound_df.bin_3.tolist()

# Split data randomly in train and test set
# note that we use test/train_x for the respective fingerprint splits
# and test/train_y for the respective label splits
(
    static_train_x_3,
    static_test_x_3,
    static_train_y_3,
    static_test_y_3,
) = train_test_split(fingerprint_to_model_3, label_to_model_3, test_size=0.2, random_state=SEED)
splits_3 = [static_train_x_3, static_test_x_3, static_train_y_3, static_test_y_3]
# NBVAL_CHECK_OUTPUT
print("Training data size:", len(static_train_x_3))
print("Test data size:", len(static_test_x_3))


# Set model parameter for random forest
param_3 = {
    "n_estimators": 100,  # number of trees to grows
    "criterion": "entropy",  # cost function to be optimized for a split
}
model_RF_3 = RandomForestClassifier(**param_3)

#change variables to _10 at end
fingerprint_to_model_10 = compound_df.maccs.tolist()
label_to_model_10 = compound_df.bin_10.tolist()

# Split data randomly in train and test set
# note that we use test/train_x for the respective fingerprint splits
# and test/train_y for the respective label splits
(
    static_train_x_10,
    static_test_x_10,
    static_train_y_10,
    static_test_y_10,
) = train_test_split(fingerprint_to_model_10, label_to_model_10, test_size=0.2, random_state=SEED)
splits_10 = [static_train_x_10, static_test_x_10, static_train_y_10, static_test_y_10]
# NBVAL_CHECK_OUTPUT
print("Training data size:", len(static_train_x_10))
print("Test data size:", len(static_test_x_10))


# Set model parameter for random forest
param_10 = {
    "n_estimators": 100,  # number of trees to grows
    "criterion": "entropy",  # cost function to be optimized for a split
}
model_RF_10 = RandomForestClassifier(**param_10)

performance_measures = model_training_and_validation(model_RF_2, "RF", splits_2)
models_RF_2 = [{"label": "RF: 2 Split", "model": model_RF_2}]
# models_RF.append({"label": "RF: 3 Split", "model": model_RF_3})
# models_RF.append({"label": "RF: 10 Split", "model": model_RF_10})
plot_roc_curves_for_models(models_RF_2, static_test_x_2, static_test_y_2)