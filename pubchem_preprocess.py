'''
REMOVE MISSING DATA, INORGANICS, MIXTURES, DUPLICATES OF TRAP1 DATASET FROM PUBCHEM DATABASE.
CALCULATE PIC50 FROM IC50.
'''
#Author TARAPONG SRISONGKRAM, Ph.D.
#Copyright.
'''
We first need to remove missing data, mixtures, and duplicate,
We will use Tokenizer, so we can use inorganics (salt); that why we will not delete that one in this stage.
We want to select only the cid, acvalue_um (microM), name, assay, and isometric_smiles
'''
#Import df_selection
import pandas as pd
def remove_missing_data(df, smiles, activity=None):
    '''
    This function remove missing data from specific dataframe and columns
    -------
    Parameters:
    df : Dataframe
    smiles: SMILES column
    activity: Activity column such as IC50
    -------
    Return:
    new df with no missing from specific columns.
    '''
    subset_cols = [smiles]
    if activity is not None and activity in df.columns:
        subset_cols.append(activity)

    df_select = df.dropna(subset=subset_cols)

    print(f"Removed {len(df) - len(df_select)} missing rows. {len(df_select)} remaining.")
    return df_select

def canonical_smiles(df, smiles_col):
    '''
    This function turn your smile to Canonical SMILES with isomeric
    ------
    Parameters:
    df : Dataframe
    smiles: SMILES column
    ------
    return new df with isomeric smiles
    '''
    #generate canonical smiles
    df['canonical_smiles'] = df[smiles_col].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True))
    df = df.drop(columns=[smiles_col], axis=1)
    print("Finish converting SMILES to canonical SMILES (isomeric)")
    return df

from rdkit.Chem import AllChem as Chem
def has_carbon_atoms(smiles):
    '''
    Helper function.
    Check whether SMILES contain carbon atoms.
    if molecule contains carbon atoms, it will return TRUE, else FALSE
    ------
    Parameters
    smiles: SMILES STRING
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        return len(carbon_atoms) > 0
    return False

def remove_inorganic(df, smiles_col):
    '''
    Remove inorganics (no carbon) using TRUE/FALSE
    Select only organics (has carbon atom == TRUE)
    ------
    Parameters:
    df: Dataframe
    smiles_col: SMILES column in dataframe (df)
    '''
    has_carbon = df[smiles_col].apply(has_carbon_atoms)
    df_select = df[has_carbon == True]
    number_row_before = len(df)
    number_row_after  = len(df_select)
    print('Remove ', str(number_row_before - number_row_after), ' INORGANICS data. Only ', number_row_after, ' remaining')
    return df_select

def remove_mixtures(df, smiles_col):
    '''
    Check if molecule is a mixture using '.' as separator and store in is_mixture.
    Drop mixture if SMILES contain . (is_mixture == FALSE)
    ------
    Parameters:
    df: Dataframe
    smiles_col: SMILES column in dataframe (df)
    '''
    is_mixture = df[smiles_col].apply(lambda x: '.' in x)
    df_select = df[is_mixture == False]
    number_row_before = len(df)
    number_row_after  = len(df_select)
    print('Remove ', str(number_row_before - number_row_after), ' MIXTURES data. Only ', number_row_after, ' remaining')
    return df_select

def select_columns(df, ligandID, smiles_col, activity_col):
    '''
    Select specific columns
    ------
    Parameters:
    df: Dataframe
    ligandID: Compound index
    smiles_col: SMILES column in df
    activity_col: IC50 column
    '''
    df_select = df[[ligandID, smiles_col, activity_col]]
    return df_select

def process_duplicate(df, smiles_col, remove_duplicate=False):
    '''
    Get duplicate from SMILES
    Average acvalues_um from SMILED
    ------
    Parameters:
    df: Dataframe
    smiles_col: SMILES column in df
    '''
    duplicate_entries = df[df.duplicated(subset=smiles_col, keep = False)].sort_values(smiles_col)
    #save duplicate for inspection
    duplicate_entries.to_csv('duplicates.csv')
    #specify
    if remove_duplicate == True:
        df_no_duplicate = df.drop_duplicates(subset=[smiles_col], keep=False)
    else:
        df_no_duplicate = df.groupby(smiles_col).mean().reset_index()
    number_duplicate = len(duplicate_entries)
    number_row_after = len(df_no_duplicate)
    print('This dataframe contains ', number_duplicate, ' duplicate entries')
    print('After remove DUPLICATES data, this dataframe contain ', number_row_after, ' data')
    return df_no_duplicate

import numpy as np
def calculate_pic50(df, ic50_col):
    '''
    Compute pIC50 (M) pIC50 = -Log(IC50) (M)
    -------
    Parameters:
    df: Dataframe
    ic50_col: IC50 columns (uM)
    '''
    #change uM to M 
    ic50_m = df[ic50_col] * 1e-6
    pic50  = -np.log10(ic50_m)
    df['pIC50'] = pic50.round(2)
    #drop IC50 column
    df = df.drop(ic50_col, axis=1)
    print('Successfully converted ic50 to pIC50')
    return df

def main():

    print("This software is starting to remove missing data, inorganics, mixtures, select columns, average duplicates, computed pIC50 from uM ..")
    smiles = str(input("Type your SMILES columns name  "))
    acvalue_um = str(input("Type your Activity columns name  "))
    cid = str(input("Type your LigandID columns name  "))
    print("#"*100)
    print("Preprocessing data")
    print("Total datapoint = ", len(df))
    df_select = remove_missing_data(df, smiles, acvalue_um)
    df_select = remove_inorganic(df_select, smiles)
    df_select = remove_mixtures(df_select, smiles)
    df_select = select_columns(df_select, cid, smiles, acvalue_um)
    df_select = process_duplicate(df_select, smiles)
    df_select = calculate_pic50(df_select, acvalue_um)
    print("Save data")
    df_select.to_csv('pubchem_no_duplicate.csv')
    print("Example of results")
    print(df_select)
    print("#"*100)
    print('Finished!')

if __name__ == "__main__":
    main()



