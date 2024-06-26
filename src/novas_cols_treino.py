
import pandas as pd
from rdkit import Chem
from rdkit.Chem import EState
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

def get_smarts(padrao, smiles):
    mol = Chem.MolFromSmiles(smiles)
    count = 0
    for j in Chem.EState.AtomTypes.TypeAtoms(mol):
        if j == (padrao,):
            count += 1
    return count 


def get_number_ligacoes_pi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pi = 0
    for i in mol.GetBonds():
        if i.GetBondType() == Chem.BondType.DOUBLE:            
            pi += 1
        elif i.GetBondType() == Chem.BondType.TRIPLE:
            pi += 2
    return pi


def get_atom_count(smiles, element):
    count = 0
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        simbolo_atom = atom.GetSymbol()
        if simbolo_atom == element:
            count += 1
    return count


def get_number_aromatic_atoms(smiles):
    count = 0
    mol = Chem.MolFromSmiles(smiles)
    number_atoms = mol.GetNumAtoms()
    for i in range(0, number_atoms):
        if mol.GetAtomWithIdx(i).GetIsAromatic():
            count += 1
    return count


def get_is_ion(smiles):
    return 1 if '+' in smiles else 0


def get_atoms_in_aromatic(smiles, element):
    count = 0
    mol = Chem.MolFromSmiles(smiles)
    number_atoms = mol.GetNumAtoms()

    for i in range(0, number_atoms):
        if mol.GetAtomWithIdx(i).GetIsAromatic():
            atomic_symbol = mol.GetAtomWithIdx(i).GetSymbol()
            if atomic_symbol == element:
                count += 1
    return count


def generate_df(target):
    df = pd.read_csv('./dados_pub.csv')  # .head(10000)

    cols = [
        'smiles', 'num_atoms', 'smarts_tN', 'smarts_aaO', 'ligacoes_pi',
        'n_atoms_O', 'n_atoms_N', 'n_atoms_C', 'n_atoms_F',
        'aromatic_atoms', 'is_ion', 'aromatic_O', 'aromatic_N',
        'aromatic_C', 'aromatic_F'
    ]
    df_treino = pd.DataFrame(columns=cols)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        nova_linha = {
            'smiles': row.smiles,
            'num_atoms': row.n_atoms,
            'smarts_tN': get_smarts('tN', row.smiles),
            'smarts_aaO': get_smarts('aaO', row.smiles),
            'ligacoes_pi': get_number_ligacoes_pi(row.smiles),
            'n_atoms_O': get_atom_count(row.smiles, 'O'),
            'n_atoms_N': get_atom_count(row.smiles, 'N'),
            'n_atoms_C': get_atom_count(row.smiles, 'C'),
            'n_atoms_F': get_atom_count(row.smiles, 'F'),
            'aromatic_atoms': get_number_aromatic_atoms(row.smiles),
            'is_ion': get_is_ion(row.smiles),
            'aromatic_O': get_atoms_in_aromatic(row.smiles, 'O'),
            'aromatic_N': get_atoms_in_aromatic(row.smiles, 'N'),
            'aromatic_C': get_atoms_in_aromatic(row.smiles, 'C'),
            'aromatic_F': get_atoms_in_aromatic(row.smiles, 'F'),
            'target': row[target]
        }
        df_treino = pd.concat([df_treino, pd.DataFrame([nova_linha])])
    df_treino.to_csv('dados_treino.csv', index=False)


generate_df('homo_lumo_nm')

