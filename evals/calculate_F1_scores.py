import pandas as pd
from sklearn.metrics import f1_score

# Read true labels
true_df = pd.read_csv("scifact_test_data.csv")
y_true = true_df["label"]

# List results files
results_files = [
    "predict_with_abstracts.csv",
    "predict_with_PDFs_k5.csv",
    "predict_with_PDFs_k10.csv",
]
# List results descriptions
results_descriptions = ["abstracts", "PDFs (k=5)", "PDFs (k=10)"]

# Loop over files and descriptions
for results_file, results_description in zip(results_files, results_descriptions):

    # Read predicted labels
    pred_df = pd.read_csv(results_file)
    y_pred = pred_df["label"]
    # Put true and predicted labels in DataFrame
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    # Drop missing values
    df.dropna(inplace=True)
    # Calculate macro F1
    f1 = f1_score(df["y_true"], df["y_pred"], average="macro")
    print(f"Macro F1 with {results_description}: {round(f1, 3)} (n={df.shape[0]})")
