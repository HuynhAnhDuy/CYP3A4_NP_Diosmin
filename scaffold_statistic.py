import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import fisher_exact
import warnings

warnings.filterwarnings("ignore")

# === 1. Đọc dữ liệu ===
input_file = "/home/andy/andy/CYP3A4_NP_Diosmin/CYP3A4_preprocess.csv"  # 👈 Cập nhật đường dẫn file của bạn
df = pd.read_csv(input_file)

# Đảm bảo tên cột chính xác
df = df.rename(columns={'canonical_smiles': 'SMILES', 'Toxicity Value': 'Label'})

# Kiểm tra cột cần thiết
assert 'SMILES' in df.columns and 'Label' in df.columns, "❌ Thiếu cột 'SMILES' hoặc 'Label'"
assert df['Label'].isin([0, 1]).all(), "❌ Cột 'Label' phải là nhị phân (0 hoặc 1)"

# ==== 2. Molecule standardization + scaffold extraction ====
def _standardize_mol(mol: Chem.Mol) -> Chem.Mol:
    """Chuẩn hoá phân tử trước khi lấy scaffold"""
    if mol is None:
        return None
    try:
        params = rdMolStandardize.CleanupParameters()
        mol = rdMolStandardize.Cleanup(mol, params)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)     # Giữ mảnh lớn nhất
        mol = rdMolStandardize.Uncharger().uncharge(mol)                # Trung hoá điện tích
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)   # Canonical tautomer
        return mol
    except Exception:
        return None

def get_scaffold(smiles: str) -> str:
    """Trích xuất Murcko scaffold đã chuẩn hoá từ SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    mol = _standardize_mol(mol)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(
            core,
            isomericSmiles=False,
            kekuleSmiles=False,
            canonical=True
        )
    except Exception:
        return None

# Áp dụng scaffold extraction
df['Scaffold'] = df['SMILES'].apply(get_scaffold)
df = df.dropna(subset=['Scaffold'])  # loại SMILES lỗi

# === 3. Đếm tần suất scaffold theo nhóm nhãn ===
carc_count = df[df['Label'] == 1]['Scaffold'].value_counts()
noncarc_count = df[df['Label'] == 0]['Scaffold'].value_counts()

scaffold_df = pd.DataFrame({
    'Inhibitor': carc_count,
    'Noninhibitor': noncarc_count
}).fillna(0)

# Tổng số mẫu mỗi nhóm
total_carc = df['Label'].sum()
total_noncarc = (df['Label'] == 0).sum()

# === 4. Tính Fisher's Exact Test và Odds Ratio ===
def fisher_test(row):
    a = int(row['Inhibitor'])
    b = int(row['Noninhibitor'])
    c = total_carc - a
    d = total_noncarc - b
    table = [[a, b], [c, d]]
    oddsratio, p = fisher_exact(table)
    return pd.Series({'OddsRatio': oddsratio, 'p_value': p})

# Áp dụng thống kê
stats = scaffold_df.apply(fisher_test, axis=1)
scaffold_df = scaffold_df.join(stats)

# (Tuỳ chọn) Lọc scaffold xuất hiện quá ít
# scaffold_df = scaffold_df[(scaffold_df['Inhibitor'] + scaffold_df['Noninhibitor']) >= 3]

# === 5. Sắp xếp và xuất kết quả ===
scaffold_df = scaffold_df.sort_values(by=['p_value', 'OddsRatio'], ascending=[True, False])
scaffold_df.to_csv("CYP3A4_scaffold_stat_analysis.csv", index=True)

print("✅ Phân tích xong. File kết quả: CYP3A4_scaffold_stat_analysis.csv")
print(f"📊 Số scaffold được phân tích: {len(scaffold_df)}")
