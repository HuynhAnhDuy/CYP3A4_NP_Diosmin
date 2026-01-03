import pandas as pd

def merge_feature_files(prefix, feature_suffixes, output_name=None):
    """
    prefix: chuỗi prefix chung, ví dụ: 'InFlam2_full_x_train'
    feature_suffixes: list các hậu tố file, ví dụ:
        ['ecfp', 'rdkit', 'maccs', 'estate', 'phychem', 'substruct']
    output_name: tên file output, nếu None sẽ là f"{prefix}_all_features.csv"
    """

    dfs = []
    for suffix in feature_suffixes:
        filename = f"{prefix}_{suffix}.csv"
        df = pd.read_csv(filename)
        dfs.append(df)
        print(f"Loaded {filename} with shape {df.shape}")

    # Gộp theo cột 'Index'
    # Dùng merge tuần tự để đảm bảo chỉ những hàng có Index chung được giữ lại
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on="Index", how="inner")

    if output_name is None:
        output_name = f"{prefix}_all_features.csv"

    merged_df.to_csv(output_name, index=False)
    print(f"✅ Saved merged features to {output_name}, shape = {merged_df.shape}")


def main():
    feature_suffixes = ["ecfp", "rdkit", "maccs", "estate", "phychem", "substruct"]

    # Train
    merge_feature_files(
        prefix="CYP3A4_external_x_train",
        feature_suffixes=feature_suffixes,
        output_name="CYP3A4_external_x_train_all_features.csv"
    )

    # Test
    merge_feature_files(
        prefix="CYP3A4_external_x_test",
        feature_suffixes=feature_suffixes,
        output_name="CYP3A4_external_x_test_all_features.csv")


if __name__ == "__main__":
    main()
