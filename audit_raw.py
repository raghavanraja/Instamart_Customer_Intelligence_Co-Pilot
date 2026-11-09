import pandas as pd
from common import RAW, ensure_dirs

def main():
    ensure_dirs()
    print("RAW DATA AUDIT")
    for path in sorted(RAW.glob("*.csv")):
        df = pd.read_csv(path)
        print(f"\n{path.name}: rows={len(df)}, columns={len(df.columns)}")
        print("columns:", ", ".join(df.columns))
        print("missing values:", df.isna().sum().to_dict())
        print("duplicate rows:", int(df.duplicated().sum()))
        print("sample:")
        print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
