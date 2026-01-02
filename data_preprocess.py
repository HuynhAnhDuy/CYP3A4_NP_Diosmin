#Import df_selection
import pandas as pd
import pubchem_preprocess as pp
from rdkit.Chem import AllChem as Chem

def main():
    df = pd.read_csv('/home/andy/andy/CYP3A4_NP_Diosmin/CYP3A4_x_external.csv')
    df_selection = pp.remove_missing_data(df,'SMILES','Label')
    df_selection = pp.canonical_smiles(df_selection,'SMILES')
    df_selection = pp.remove_inorganic(df_selection,'canonical_smiles')
    df_selection = pp.remove_mixtures(df_selection,'canonical_smiles')
    df_selection = pp.process_duplicate(df_selection,'canonical_smiles',remove_duplicate=True)
    df_selection.to_csv('CYP3A4_x_external_preprocess.csv')
    print("Finished!")

main()