import pandas as pd

xlsx_path = "Dataset_FOVLeaks.xlsx"

xls = pd.ExcelFile(xlsx_path)
print("Sheets:", xls.sheet_names)

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df["Label_in_Seq"] = df.apply(
        lambda r: int(str(r["Label"]) in str(r["Sequence"])),
        axis=1
    )
    
    result = df.groupby("Label").agg(
        seq_hit_rate=("Label_in_Seq", "mean"),
        count=("Label", "count")
    ).reset_index()


    result["seq_hit_rate"] = (result["seq_hit_rate"] * 100).round(2)

    print(f"\n=== Sheet: {sheet_name} ===")
    print(result)
