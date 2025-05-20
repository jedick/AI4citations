import os
import sys
import pandas as pd
from gradio_client import Client, handle_file

# Script to make predictions on SciFact test data with evidence retrieval from PDFs

# Start gradio client
client = Client("http://127.0.0.1:7860/")

# Read SciFact test data
df = pd.read_csv("scifact_test_data.csv")

# Initialize lists for labels and evidences
labels = []
evidences = []

# Set top k sentences for retrieval
top_k = 5

# Loop over examples
for index, row in df.iterrows():

    # Show progress
    if index % 10 == 0:
        print(index)

    # Get PDF file for this claim
    pdf_file = f"/home/jedick/tmp/scifact-test-pdfs/{row['corpus_id']}.pdf"
    if not os.path.exists(pdf_file):

        # Append empty results if PDF doesn't exist
        labels.append("")
        evidences.append("")

    else:

        # Get evidence from PDF
        try:
            evidence = client.predict(
                pdf_file=handle_file(pdf_file),
                query=row["claim"],
                k=top_k,
                api_name="/retrieve_from_pdf",
            )
            evidences.append(evidence)
        except:
            # If an error occured, print the file name and error message
            error_type, error_value, error_traceback = sys.exc_info()
            print(pdf_file)
            print(error_type)
            print(error_value)
            # Use empty label and skip classification step
            labels.append("")
            evidences.append("")
            continue

        # Predict the classifiction
        result = client.predict(
            claim=row["claim"], evidence=evidence, api_name="/query_model"
        )
        label = result[1]["label"]
        labels.append(label)

# Append top-k value to output file name
output_file = f"predict_with_PDFs_k{top_k}.csv"

## For development and monitoring only:
## (We can't upload the evidences retrieved from PDFs because they might contain copyrighted material)
## Convert labels and evidences to DataFrame and save as CSV
#results_df = pd.DataFrame(zip(labels, evidences), columns=["label", "evidence"])
#results_df.to_csv(output_file, index=False)

# For uploading test results:
results_df = pd.DataFrame(labels, columns=["label"])
results_df.to_csv(output_file, index=False)
