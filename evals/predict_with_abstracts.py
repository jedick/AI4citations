import pandas as pd
from gradio_client import Client

# Script to make predictions on SciFact test data with abstracts as evidence

# Read SciFact test data
df = pd.read_csv("scifact_test_data.csv")
labels = []

# Start client
client = Client("http://127.0.0.1:7860/")

for index, row in df.iterrows():
    if index % 10 == 0:
        print(index)
    result = client.predict(
        claim=row["claim"], evidence=row["abstract"], api_name="/query_model"
    )
    label = result[1]["label"]
    labels.append(label)

# Convert labels to DataFrame and save as CSV
labels_df = pd.DataFrame(labels, columns=["label"])
labels_df.to_csv("predict_with_abstracts.csv", index=False)
