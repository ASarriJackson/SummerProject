from pathlib import Path
import sys
# Get the directory of the current file
current_file_path = Path(__file__)

# Get the parent directory of the current directory (i.e., SummerProject)
parent_directory = current_file_path.parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_directory))
from helper_fun import *
from tqdm import tqdm
import seaborn as sns
import umap
import sklearn.datasets
import plotly.express as px
SEED = 42
seed_everything(SEED)

#from OPIG website
def fingerprint_list_from_smiles_list(smiles_list, n_bits=2048):
    fingerprint_list = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        fingerprint_list.append(fingerprint_as_array(mol, n_bits))
    return fingerprint_list

def fingerprint_as_array(mol, n_bits=2048):
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    array = np.zeros((1,), int)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array

def get_umap_fingerprint_array(table, smiles_column="SMILES"):
    SMILES_list = [x for x in table.smiles_column]
    fingerprint_list = fingerprint_list_from_smiles_list(SMILES_list)
    fingerprint_array = np.array(fingerprint_list) 
    return fingerprint_array

def get_umap_fingerprint_array_fig(table, CID_column="CID", pIC50_column="f_avg_pIC50"):
    fingerprint_array = get_umap_fingerprint_array(table)
    umap_reducer = umap.UMAP()
    umap_fingerprint_array = umap_reducer.fit_transform(fingerprint_array)
    umap_fingerprint_array_fig = pd.DataFrame(umap_fingerprint_array, columns=["X","Y"])
    umap_fingerprint_array_fig[CID_column] = table[CID_column].values
    umap_fingerprint_array_fig[pIC50_column] = table[pIC50_column].values
    umap_fingerprint_array_fig.dropna(subset=[CID_column], inplace=True)
    return umap_fingerprint_array_fig

def get_umap_fingerprint_plot(table, CID_column="CID", pIC50_column="f_avg_pIC50"):
    custom_data = umap_fingerprint_array_fig[[CID_column, pIC50_column]]

    df_fig = px.scatter(umap_fingerprint_array_fig, x="X", y="Y",
                    hover_data= [CID_column, pIC50_column],
                    custom_data= custom_data,
                    title="UMAP Projection of Molecules")

    df_fig.update_traces(hovertemplate='CID: %{customdata[0]}<br>f_avg_pIC50: %{customdata[1]}')
    df_fig.update_layout(width=800, height=800, transition_duration=500)

    return df_fig.show()

#edit below
#define a function called split_furthest cluster that inputs the umap_fingerprint_array_fig and outputs X_train, X_test, Y_train, Y_test where the test set is a cluster of 20% of the data that is least like the other 80% of the data
def split_furthest_cluster(table, CID_column="CID", pIC50_column="f_avg_pIC50", test_size=0.2):
    umap_fingerprint_array_fig = get_umap_fingerprint_array_fig(table)
    



