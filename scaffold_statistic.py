import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import fisher_exact
import warnings

warnings.filterwarnings("ignore")

# === 1. Đọc dữ liệu ===
input_file = "capsule_x_train.csv"  # 👈 THAY bằng tên file bạn có
df = pd.read_csv(input_file)

# Đảm bảo tên cột chính xác
df = df.rename(columns={'canonical_smiles': 'SMILES', 'Toxicity Value': 'Label'})

# === 2. Trích xuất scaffold ===
def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None

df['Scaffold'] = df['SMILES'].apply(get_scaffold)
df = df.dropna(subset=['Scaffold'])  # loại những dòng lỗi SMILES

# === 3. Đếm tần suất theo Label ===
carc_count = df[df['Label'] == 1]['Scaffold'].value_counts()
noncarc_count = df[df['Label'] == 0]['Scaffold'].value_counts()

scaffold_df = pd.DataFrame({
    'Carcinogen': carc_count,
    'Non_Carcinogen': noncarc_count
}).fillna(0)

# Tổng số mẫu theo nhóm
total_carc = df['Label'].sum()
total_noncarc = (df['Label'] == 0).sum()

# === 4. Tính Fisher's Exact Test và Odds Ratio ===
def fisher_test(row):
    a = int(row['Carcinogen'])
    b = int(row['Non_Carcinogen'])
    c = total_carc - a
    d = total_noncarc - b
    table = [[a, b], [c, d]]
    oddsratio, p = fisher_exact(table)
    return pd.Series({'OddsRatio': oddsratio, 'p_value': p})

stats = scaffold_df.apply(fisher_test, axis=1)
scaffold_df = scaffold_df.join(stats)

# === 5. Sắp xếp và xuất kết quả ===
scaffold_df = scaffold_df.sort_values(by='p_value')
scaffold_df.to_csv("scaffold_stat_analysis.csv", index=True)

print("✅ Phân tích xong. File kết quả: scaffold_stat_analysis.csv")
