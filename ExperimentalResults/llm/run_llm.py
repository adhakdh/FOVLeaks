import pandas as pd

df = pd.ExcelFile("Dataset_FOVLeaks.xlsx")

for sheet_name in df.sheet_names:
    if sheet_name == "password":
        continue
    df = pd.read_excel("Dataset_FOVLeaks.xlsx", sheet_name=sheet_name)
    input_Label = "Label"
    if sheet_name == "web":
        input_Label = "Label2"

    df["Top1_correct"] = (df[input_Label] == df["Top-1"]).astype(int)

    result = df.groupby("Label").agg(
        top1_acc=("Top1_correct", "mean"),
    ).reset_index()


    result["top1_acc"] = (result["top1_acc"] * 100).round(2)

    print(f"\n=== Sheet: {sheet_name} ===")
    print(result)
