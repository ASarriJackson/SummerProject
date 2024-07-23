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

def get_umap_fingerprint_array(table):
    SMILES_list = [x for x in table.SMILES]
    fingerprint_list = fingerprint_list_from_smiles_list(SMILES_list)
    fingerprint_array = np.array(fingerprint_list) 
    return fingerprint_array

def get_umap_fingerprint_plot(table):
    fingerprint_array = get_umap_fingerprint_array(table)
    umap_reducer = umap.UMAP()
    umap_fingerprint_array = umap_reducer.fit_transform(fingerprint_array)
    umap_fingerprint_array_fig = pd.DataFrame(umap_fingerprint_array, columns=["X","Y"])
    umap_fingerprint_array_fig['CID'] = table['CID'].values
    umap_fingerprint_array_fig['f_avg_pIC50'] = table['f_avg_pIC50'].values
    umap_fingerprint_array_fig.dropna(subset=['CID'], inplace=True)
    custom_data = umap_fingerprint_array_fig[['CID', 'f_avg_pIC50']]

    df_fig = px.scatter(umap_fingerprint_array_fig, x="X", y="Y",
                    hover_data= ['CID', 'f_avg_pIC50'],
                    custom_data= custom_data,
                    title="UMAP Projection of Molecules")

    df_fig.update_traces(hovertemplate='CID: %{customdata[0]}<br>f_avg_pIC50: %{customdata[1]}')
    df_fig.update_layout(width=800, height=800, transition_duration=500)

    return df_fig.show()

