import sys
sys.path.append('..')
sys.path.append('/home/alex/SummerProject')
from helper_fun import *
from optuna.samplers import TPESampler
from Split_functions_classify.splits_two import *
import joblib


SEED = 42
seed_everything(SEED)

compounds = pd.read_csv('COVID_MOONSHOT/compounds_filtered.csv')
two_split(compounds)
compound_df=compounds.copy()
compound_df["maccs"] = compound_df["SMILES"].apply(smiles_to_fp)

fingerprint_to_model = compound_df.maccs.tolist()
label_to_model_2 = compound_df.bin_2.tolist()

# Split data randomly in train and test set
# note that we use test/train_x for the respective fingerprint splits
# and test/train_y for the respective label splits
(
    static_train_x_2,
    static_test_x_2,
    static_train_y_2,
    static_test_y_2,
) = train_test_split(fingerprint_to_model, label_to_model_2, test_size=0.2, random_state=SEED)
splits_2 = [static_train_x_2, static_test_x_2, static_train_y_2, static_test_y_2]

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
    model.fit(static_train_x_2, static_train_y_2)
    y_pred = model.predict(static_test_x_2)
    y_pred_proba = model.predict_proba(static_test_x_2)[:, 1]
    return f1_score(static_test_y_2, y_pred), roc_auc_score(static_test_y_2, y_pred_proba), matthews_corrcoef(static_test_y_2, y_pred)

study = optuna.create_study(directions=["maximize","maximize","maximize"],sampler=TPESampler(seed=SEED))
study.optimize(objective, n_trials=1000, timeout=600, n_jobs=8)

best_trials = study.best_trials

values = [trial.values for trial in best_trials]
sum_of_squares = [sum([x**2 for x in trial]) for trial in values]
max_index = sum_of_squares.index(max(sum_of_squares))
best_param_RF_2 = best_trials[max_index].params

model_RF_2 = RandomForestClassifier(random_state=42)
model_RF_2.set_params(**best_param_RF_2)

joblib_file = "Classifiers/Optuna_model_RF_2.pkl"
joblib.dump(model_RF_2, joblib_file)