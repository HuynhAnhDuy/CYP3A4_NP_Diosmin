import pandas as pd
import numpy as np
from itertools import chain

from rdkit import Chem
import selfies as sf

from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================== 1. Kiểm tra SMILES hợp lệ ================== #

def is_valid_smiles(smiles: str) -> bool:
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


# ================== 2. SMILES → SELFIES ================== #

def smiles_to_selfies(smiles: str) -> str:
    """
    Convert SMILES → SELFIES.
    Trả về None nếu SMILES không hợp lệ hoặc encode thất bại.
    """
    if not is_valid_smiles(smiles):
        return None
    try:
        # có thể canonical hóa lại nếu cần
        mol = Chem.MolFromSmiles(smiles)
        can_smi = Chem.MolToSmiles(mol)
        selfies_str = sf.encoder(can_smi)
        return selfies_str
    except Exception:
        return None


# ================== 3. SELFIES → tokens ================== #

def selfies_to_tokens(selfies_str: str):
    """
    SELFIES string → list token, vd: '[C][C][=O][O]' → ['[C]', '[C]', '[=O]', '[O]']
    """
    if selfies_str is None:
        return []
    try:
        return list(sf.split_selfies(selfies_str))
    except Exception:
        return []


# ================== 4. Xây vocab từ token (chỉ dùng TRAIN) ================== #

def build_vocab_from_tokens(list_of_token_lists):
    """
    list_of_token_lists: danh sách các sequence, mỗi sequence là list các token SELFIES
    Trả về:
        - vocab_to_idx: dict {token: id}
        - idx_to_vocab: dict {id: token}
    """
    all_tokens = list(chain.from_iterable(list_of_token_lists))
    unique_tokens = sorted(set(all_tokens))

    # id = 0 dùng cho PAD
    vocab_to_idx = {tok: i + 1 for i, tok in enumerate(unique_tokens)}
    idx_to_vocab = {i: tok for tok, i in vocab_to_idx.items()}

    return vocab_to_idx, idx_to_vocab


def tokens_to_ids(tokens, vocab_to_idx):
    return [vocab_to_idx[t] for t in tokens if t in vocab_to_idx]


# ================== 5. Hàm chính: CSV → sequence padded ================== #

def prepare_selfies_bilstm_input(
    train_csv_path: str,
    test_csv_path: str,
    smiles_col: str = "canonical_smiles",
    max_len: int = None,
):
    """
    - Đọc x_train, x_test từ CSV
    - Chuyển SMILES → SELFIES → tokens
    - Xây vocab từ TRAIN
    - Encode tokens → IDs
    - Pad sequence cho BiLSTM

    Return:
        X_train_seq, X_test_seq, vocab_to_idx, idx_to_vocab, max_len
    """

    # ----- Load dữ liệu ----- #
    x_train = pd.read_csv(train_csv_path, index_col=0)
    x_test  = pd.read_csv(test_csv_path, index_col=0)

    # ----- SMILES → SELFIES ----- #
    x_train["SELFIES"] = x_train[smiles_col].apply(smiles_to_selfies)
    x_test["SELFIES"]  = x_test[smiles_col].apply(smiles_to_selfies)

    # Có thể loại bỏ dòng không encode được SELFIES (nếu cần)
    x_train = x_train[~x_train["SELFIES"].isna()].copy()
    x_test  = x_test[~x_test["SELFIES"].isna()].copy()

    # ----- SELFIES → tokens ----- #
    x_train["SELFIES_tokens"] = x_train["SELFIES"].apply(selfies_to_tokens)
    x_test["SELFIES_tokens"]  = x_test["SELFIES"].apply(selfies_to_tokens)

    # ----- Xây vocab từ TRAIN ----- #
    vocab_to_idx, idx_to_vocab = build_vocab_from_tokens(
        x_train["SELFIES_tokens"].tolist()
    )

    # ----- Encode tokens → IDs ----- #
    x_train["SELFIES_ids"] = x_train["SELFIES_tokens"].apply(
        lambda toks: tokens_to_ids(toks, vocab_to_idx)
    )
    x_test["SELFIES_ids"] = x_test["SELFIES_tokens"].apply(
        lambda toks: tokens_to_ids(toks, vocab_to_idx)
    )

    # ----- Chọn max_len ----- #
    if max_len is None:
        # ví dụ: dùng percentile 95 chiều dài train
        lengths = x_train["SELFIES_ids"].apply(len).values
        if len(lengths) == 0:
            raise ValueError("Train set không có SELFIES hợp lệ.")
        max_len = int(np.percentile(lengths, 95))
        max_len = max(max_len, 10)  # tránh max_len quá nhỏ

    # ----- Padding cho BiLSTM ----- #
    X_train_seq = pad_sequences(
        x_train["SELFIES_ids"],
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=0,   # PAD token id
    )
    X_test_seq = pad_sequences(
        x_test["SELFIES_ids"],
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=0,
    )

    return X_train_seq, X_test_seq, vocab_to_idx, idx_to_vocab, max_len, x_train, x_test


# ================== 6. Ví dụ sử dụng ================== #

if __name__ == "__main__":
    train_csv = "CYP3A4_x_train.csv"
    test_csv  = "CYP3A4_x_test.csv"

    X_train_seq, X_test_seq, vocab2idx, idx2vocab, max_len, x_train, x_test = \
        prepare_selfies_bilstm_input(
            train_csv_path=train_csv,
            test_csv_path=test_csv,
            smiles_col="canonical_smiles",
            max_len=None,
        )

    # Lưu sequence thành CSV
    train_seq_df = pd.DataFrame(X_train_seq, index=x_train.index)
    test_seq_df  = pd.DataFrame(X_test_seq, index=x_test.index)

    train_seq_df.to_csv("CYP3A4_x_train_selfies.csv")
    test_seq_df.to_csv("CYP3A4_x_test_selfies.csv")

    # Lưu vocab thành CSV hoặc JSON
    vocab_df = pd.DataFrame(
        [(tok, idx) for tok, idx in vocab2idx.items()],
        columns=["token", "id"]
    )
    vocab_df.to_csv("CYP3A4_selfies_vocab.csv", index=False)

    print("Saved:")
    print("  CYP3A4_x_train_selfies.csv")
    print("  CYP3A4_x_test_selfies.csv")
    print("  CYP3A4_selfies_vocab.csv")
    print("X_train_seq shape:", X_train_seq.shape)
    print("X_test_seq shape:", X_test_seq.shape)
    print("Vocab size:", len(vocab2idx))
    print("Max sequence length:", max_len)
