import pandas as pd   
import numpy as np  
from sklearn.feature_selection import VarianceThreshold
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors  
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class DataIngestion():
    
    '''
    Class for handling the data of dataset QM9

    Atributes: 
    data: data with smiles column and targets columns, like homo-lumo and dipole moment.
    name_column_smile: It is the name of smiles column, to generalize the analysis.

    Methods: 
        drop_duplicated: drop duplicated smiles.

        get_features: add features to dataset.

        get_best_features: chooses the best features from features of "get_features" method.

    '''

    def drop_duplicated(self, data, name_column_smiles="smiles", load=False ):

        '''
        This methods drop lines, whose molecule is duplicated. Because of simetry diferents smiles can represent 
    the same molecules.

        Args: 
            data (pandas.Dataframe): Input unprocessed data.
            name_column_smiles (str: default = "smiles"): It is the name of smiles column, to generalize the analysis. 
            
        Return:
            data (pandas.Dataframe): Output processed data.
        '''

        # Create a list of all canonical smiles
        mols = [Chem.MolFromSmiles(smi) for smi in data[name_column_smiles]] 
        canonical_smiles = [Chem.MolToSmiles(mol) for mol in mols]

        #Update the column smile on data
        data.loc[:,name_column_smiles] = canonical_smiles

        # drop duplicated smiles
        data_new = data.drop_duplicates(subset=[name_column_smiles])

        if load:
            return data_new
        else:
            self.data_processed_1 = data_new
            return self
        
    def get_features(self, data, name_column_smiles="smiles", load=False):

        '''
        This methods add new columns to dataset, through of RDKit descriptors. 
    will be calculated all of descriptors using smiles. This is utils for to become possible others 
    analisys.

        Args: 
            data (pandas.Dataframe): Input unprocessed data.
            name_column_smiles (str: default = "smiles"): It is the name of smiles column, to generalize the analysis.
            load (bool: default = False): If True, the data will be saved in .csv format to path "data/data_feature.csv".
            
        Return:
            data (pandas.Dataframe): Output processed data in .csv format to path "data/data_features.csv", if save = True.
        '''

        if data is None:
            data = self.data_processed_1
            name_column_smiles = self.name_column_smiles

        mols = [Chem.MolFromSmiles(i) for i in data[name_column_smiles]] 

        # Gera uma lista dos descritores em objetos
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] 
                                    for x in Descriptors._descList])
    
        # Nome dos descritores
        desc_names = calc.GetDescriptorNames()

        Mol_descriptors = []

        for mol in mols:
            # add hydrogens to molecules
            mol=Chem.AddHs(mol)
            # Gera uma lista dos valores dos descritores
            descriptors = calc.CalcDescriptors(mol)
            # Gera uma matriz com os dados dos descritores para cada molécula
            Mol_descriptors.append(descriptors)

        df_features = pd.DataFrame(Mol_descriptors, columns=desc_names)

        df = pd.concat([data, df_features], axis=1)

        if load:
            df.to_csv("/home/edmurcn/Documentos/MeusProjetos/Predicao_Molecular/data/data_features.csv", index=False)

        self.data_features = df_features
        self.data_processed_2 = df

        return self
    
    def get_best_features(self, data=None, threshold_correlation=0.9, threshold_variance=0.1):
    
        '''
        This methods provide the best features of "get_features" methods, using correlation analisys and 
    based on variance.

        Args:
            data (pandas.DataFrame): Input the dataset of features provided on "get_features" without columns of "smiles" and target columns.
            threshold_correlation (float [0,1]: default = 0.9): Input the maximum feature absolute correlation value to keep the feature in the dataset.
            threshold_variance (float: default = 0.1): Input the minimun feature variance value to keep the feature in the dataset.
            load (bool: default = False): If True, the data will be saved in .csv format to path "data/data_feature.csv".
        Return:
            data (pandas.Dataframe): Output processed data together with the best features in .csv format to path "data/data_best_features.csv".
        '''
    
        if data is None:
            data = self.data_features

        # Below are the process of to drop highly correlated features

        # Calculate absolute correlation
        correlated_matrix = data.corr().abs()

        '''
        Upper triangle of correlation matrix

        this operation transform the data below the main diagonal of the correlated matrix into Nan;
        therefor, only one correlation between the features are analyzed.
        '''

        upper_triangle = correlated_matrix.where(np.triu(np.ones(correlated_matrix.shape),k=1).astype(bool))

        # Identify columns that have above 0.9 values of correlation
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] >= threshold_correlation)]

        # Drop columns identified
        data_dropped_correlated = data.drop(columns=to_drop, axis=1)

        # Below are the process of to drop feature with slight variance

        # Using the scikit-learn module "VarianceThreshold"
        selection = VarianceThreshold(threshold_variance)
        selection.fit(data_dropped_correlated)

        # Apply the selection columns in data
        data_dropped_variance = data_dropped_correlated[data_dropped_correlated.columns[selection.get_support(indices=True)]]

        '''
        Create a new dataset with selected features

        Remember: data_processed_1 is a data unprocessed without duplicated smiles
        '''
        
        data_processed_3 = pd.concat([self.data_processed_1, data_dropped_variance], axis=1)

        data_processed_3.to_csv("/home/edmurcn/Documentos/MeusProjetos/Predicao_Molecular/data/data_best_features.csv", index=False)
        
        return self
    
    def execute_pipeline(self):

        '''
        These methods run the instance of all previous methods as a pipeline.
        
        Return:
            data (pandas.DataFrame): Return a dataset in .csv format to the path "data/data_best_features.csv", this dataset will be create applying the
        methods: drop_duplicated() / get_features() / get_best_features(), in pipeline format.

        '''
        self.drop_duplicated().get_features().get_best_features()

class DataIngestion_pipe(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self
    
    def transform(self, X, y=None, name_column_smiles="smiles"):

        # Create a list of all canonical smiles
        mols = [Chem.MolFromSmiles(smi) for smi in X[name_column_smiles]] 
        canonical_smiles = [Chem.MolToSmiles(mol) for mol in mols]

        #Update the column smile on data
        X.loc[:,name_column_smiles] = canonical_smiles

        # drop duplicated smiles
        X_drop = X.drop_duplicates(subset=[name_column_smiles])
        
        # Add new features on dataset 
        mols = [Chem.MolFromSmiles(i) for i in X_drop[name_column_smiles]] 

        # Create a list with objects descriptors
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] 
                                    for x in Descriptors._descList])
    
        # Descriptors name
        desc_names = calc.GetDescriptorNames()

        Mol_descriptors = []

        for mol in mols:
            # add hydrogens to molecules
            mol=Chem.AddHs(mol)
            # Create a list with the descriptors values
            descriptors = calc.CalcDescriptors(mol)
            # Create a matrix with the descriptors values for each molecule
            Mol_descriptors.append(descriptors)

        # Create a pandas.DataFrame with the descriptors values in line and descriptors name on column name
        new_features = pd.DataFrame(Mol_descriptors, columns=desc_names)

        # Processed the new features and select the best ones

        correlated_matrix = new_features.corr().abs()

        '''
        Upper triangle of correlation matrix

        this operation transform the data below the main diagonal of the correlated matrix into Nan;
        therefor, only one correlation between the features are analyzed.
        '''

        upper_triangle = correlated_matrix.where(np.triu(np.ones(correlated_matrix.shape),k=1).astype(bool))

        # Identify columns that have above 0.9 values of correlation
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] >= 0.9)]

        # Drop columns identified
        data_dropped_correlated = new_features.drop(columns=to_drop, axis=1)

        # Below are the process of to drop feature with slight variance

        # Create a standardized dataset to analyze variation
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(data_dropped_correlated)
        data_dropped_correlated_scaled = pd.DataFrame(X_scaled, columns=data_dropped_correlated.columns)

        # Using the scikit-learn module "VarianceThreshold", with threshold=0.1
        selection = VarianceThreshold(0.1)
        selection.fit(data_dropped_correlated_scaled)

        # Apply the selection columns in data
        data_dropped_variance = data_dropped_correlated[data_dropped_correlated_scaled.columns[selection.get_support()]]


        '''
        Create a new dataset with selected features

        Remember: X_drop is a data unprocessed without duplicated smiles
        '''

        
        X_best_features = pd.concat([X_drop.reset_index(drop=True), data_dropped_variance], axis=1)

        X_best_features.to_csv("/home/edmurcn/Documentos/MeusProjetos/Predicao_Molecular/data/data_best_features.csv", index=False)
        
        return X_best_features


class PCA_pipe(BaseEstimator, TransformerMixin):

    '''
    A transformer class to reduce dimension via PCA methods

    Atributes: 
        n_components: 
       
    '''
    
    def __init__(self, n_components):

        self.n_components = n_components

    def fit(self, x_train, x_test):
        '''
        Args:
            x_train (array-like of shape (n_samples, n_features)): Training data, where n_samples is the number of training samples and n_features is the number of features.
        '''
        return self

    def transform(self, x_train, x_test):

        pca = PCA(n_components=self.n_components)

        self.x_train_reduced = pca.fit_transform(self.x_train)
        self.x_test_reduced = pca.transform(self.x_test)

        return self.x_train_reduced, self.x_test_reduced
    
