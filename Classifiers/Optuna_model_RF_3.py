import sys
sys.path.append('..')
sys.path.append('/home/alex/SummerProject')
from helper_fun import *
from optuna.samplers import TPESampler
from Split_functions_classify.splits_three import *
import joblib


SEED = 42
seed_everything(SEED)

compounds = pd.read_csv('COVID_MOONSHOT/compounds_filtered.csv')
three_split(compounds)
compound_df=compounds.copy()
compound_df["maccs"] = compound_df["SMILES"].apply(smiles_to_fp)

fingerprint_to_model = compound_df.maccs.tolist()
label_to_model_3 = compound_df.bin_3.tolist()

# Split data randomly in train and test set
# note that we use test/train_x for the respective fingerprint splits
# and test/train_y for the respective label splits
(
    static_train_x_3,
    static_test_x_3,
    static_train_y_3,
    static_test_y_3,
) = train_test_split(fingerprint_to_model, label_to_model_3, test_size=0.2, random_state=SEED)
splits_3 = [static_train_x_3, static_test_x_3, static_train_y_3, static_test_y_3]

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2",None])
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    model.fit(static_train_x_3, static_train_y_3)
    y_pred = model.predict(static_test_x_3)
    return f1_score(static_test_y_3, y_pred,average="micro"), calculate_micro_auc(model,static_test_x_3,static_train_y_3,static_test_y_3), matthews_corrcoef(static_test_y_3, y_pred)

study = optuna.create_study(directions=["maximize","maximize","maximize"],sampler=TPESampler(seed=SEED))
study.optimize(objective, n_trials=10000, timeout=900, n_jobs=8)

best_trials = study.best_trials

values = [trial.values for trial in best_trials]
sum_of_squares = [sum([x**2 for x in trial]) for trial in values]
max_index = sum_of_squares.index(max(sum_of_squares))
best_param_RF_3 = best_trials[max_index].params

model_RF_3 = RandomForestClassifier(random_state=42)
model_RF_3.set_params(**best_param_RF_3)

joblib_file = "Classifiers/Optuna_model_RF_3.pkl"
joblib.dump(model_RF_3, joblib_file)