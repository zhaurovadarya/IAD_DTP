import pandas as pd
import numpy as np
import glob

def main():
    csv_files = glob.glob("output_*.csv")
    feat_dfs = {}
    for file in csv_files:
        name = file.replace("output_", "").replace(".csv", "")
        feat_dfs[name] = pd.read_csv(file)
    try:
        geo_rand = pd.read_csv("geo_rand.csv")
    except FileNotFoundError:
        geo_rand = pd.DataFrame()
    df_rules = geo_rand.copy() if not geo_rand.empty else pd.DataFrame()
    for name, feat_df in feat_dfs.items():
        if 'feature' not in feat_df.columns:
            if 'Признак' in feat_df.columns:
                feat_df.rename(columns={'Признак': 'feature'}, inplace=True)
            else:
                feat_df.rename(columns={feat_df.columns[0]: 'feature'}, inplace=True)
        new_cols = {
            f"{name}_{f}": pd.NA
            for f in feat_df['feature'].unique()
            if f"{name}_{f}" not in df_rules.columns}
        if new_cols:
            df_rules = pd.concat([df_rules, pd.DataFrame(new_cols, index=df_rules.index)], axis=1)
    if 'severity_score' not in df_rules.columns:
        df_rules['severity_score'] = np.random.randint(0, 4, len(df_rules)) if not df_rules.empty else pd.Series(dtype='float')
    df_rules['severity_class'] = pd.cut(
        df_rules['severity_score'],
        bins=[-1, 1, 3, df_rules['severity_score'].max() if len(df_rules) > 0 else 5],
        labels=['легкая', 'средняя', 'тяжелая'])

    df_rules.to_csv("df_rules.csv", index=False, encoding="utf-8-sig")
    print("Признаки для методов сопоставления сохранены (df_rules.csv).")
    print(df_rules.head())
    return df_rules

if __name__ == "__main__":
    main()
