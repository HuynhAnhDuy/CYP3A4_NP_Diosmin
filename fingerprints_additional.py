'''Huynh Anh Duy, PharmD
Department of Health Sciences, College of Natural Sciences, Can Tho University, Viet Nam'''


import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.EState import Fingerprinter

# Function to calculate E-state indices and encode them as a fingerprint
def calculate_estate(canonical_smiles):
    """Calculates E-state indices and encodes them as a fingerprint vector."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol:
        try:
            # Calculate E-State indices (descriptors)
            estate_indices = Fingerprinter.FingerprintMol(mol)[0]
            estate_fp = [round(value, 3) for value in estate_indices]
            return estate_fp
        except Exception as e:
            print(f"Error calculating E-state indices for SMILES '{canonical_smiles}': {e}")
            return [None] * 79  # Return a vector of None for invalid cases
    return [None] * 79

# Function to compute Klekota-Roth fingerprints
def calculate_klekota_roth(canonical_smiles):
    """Calculates Klekota-Roth fingerprints (4860 bits) for a given canonical SMILES."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol:
        try:
            # Calculate Klekota-Roth fingerprints using a correct approach
            # GetKlekotaRothFingerprintAsBitVect is known to return a bit vector for fingerprints
            kr_fingerprint = rdMolDescriptors.GetKlekotaRothFingerprintAsBitVect(mol, 4860)
            return list(kr_fingerprint)  # Convert bit vector to list
        except Exception as e:
            print(f"Error calculating Klekota-Roth fingerprint for SMILES '{canonical_smiles}': {e}")
            return [None] * 4860
    return [None] * 4860

# Function to calculate PubChem fingerprints
def calculate_pubchem(canonical_smiles):
    """Calculates PubChem-like fingerprints (881 bits) for a given canonical SMILES."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol:
        try:
            # Calculate PubChem-like fingerprints
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=881)
            return list(fingerprint)
        except Exception as e:
            print(f"Error calculating PubChem fingerprint for SMILES '{canonical_smiles}': {e}")
            return [None] * 881
    return [None] * 881

# Function to calculate substructure fingerprints
def calculate_substructure(canonical_smiles, radius=2, n_bits=307):
    """Calculates substructure fingerprints (Morgan, customizable size) for a given canonical SMILES."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol:
        try:
            # Calculate Morgan substructure fingerprints
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return list(fingerprint)
        except Exception as e:
            print(f"Error calculating substructure fingerprint for SMILES '{canonical_smiles}': {e}")
            return [None] * n_bits
    return [None] * n_bits

# Function to expand and save fingerprints sequentially
def extract_and_save_fingerprints(df, output_file_prefix):
    """Calculates and saves each fingerprint type sequentially with an Index column."""
    fingerprint_types = [
        ('estate', calculate_estate, 79),
        ('klekota_roth', calculate_klekota_roth, 4860),
        ('pubchem', calculate_pubchem, 881),
        ('substructure', calculate_substructure, 307)
    ]

    for name, func, size in fingerprint_types:
        print(f"Calculating and saving {name.replace('_', ' ')}...")
        try:
            df[name] = df['canonical_smiles'].apply(func)
            fp_df = pd.DataFrame(df[name].tolist(), columns=[f'{name}_{i+1}' for i in range(size)])

            # Insert Index column from original dataframe
            fp_df.insert(0, 'Index', df.index)

            # Save to CSV
            fp_df.to_csv(f"{output_file_prefix}_{name}.csv", index=False)
        except Exception as e:
            print(f"Error processing {name}: {e}")

# Main function
def run_for_dataset(input_file, output_file_prefix):
    """Load dataset and extract/save all fingerprints."""
    print(f"\n📂 Processing {input_file}...")
    try:
        df = pd.read_csv(input_file, index_col=0)
    except FileNotFoundError:
        print(f"❌ Error: The file '{input_file}' was not found.")
        return
    except Exception as e:
        print(f"❌ Error reading '{input_file}': {e}")
        return

    extract_and_save_fingerprints(df, output_file_prefix)
    print(f"✅ Finished processing for {output_file_prefix}.\n")

def main():
    # Chạy cho x_train
    run_for_dataset('capsule_x_train.csv', 'irac_2b_x_train')

    # Chạy cho x_test
    run_for_dataset('irac_2b.csv', 'irac_2b_x_test')

if __name__ == "__main__":
    main()
