import os
import sys
import time
import pandas as pd
from gradio_client import Client, handle_file

# Script to make predictions on SciFact test data with evidence retrieval from PDFs

# Start gradio client
client = Client("http://127.0.0.1:7860/")

# Read SciFact test data
df = pd.read_csv("scifact_test_data.csv")
# Get number of examples
n = df.shape[0]

# Initialize lists for labels, evidences, and retrieval time
labels = [""] * n
evidences = [""] * n
retrieval_time = [0.0] * n
prompt_tokens = [0] * n
completion_tokens = [0] * n

# Set top k sentences for retrieval
top_k = 5
# Set retrieval method: one of ['BM25S', 'LLM (Large)', 'LLM (Fast)', 'GPT']
method = "GPT"

# Loop over examples
for index, row in df.iterrows():

    # Show progress
    if index % 10 == 0:
        print(index)

    # Get PDF file for this claim
    pdf_file = f"/home/jedick/tmp/scifact-test-pdfs/{row['corpus_id']}.pdf"
    # Only run if PDF file exists
    if os.path.exists(pdf_file):

        # Get evidence from PDF
        try:
            # Measure retrieval time
            start_time = time.time()

            evidence, _prompt_tokens, _completion_tokens = client.predict(
                pdf_file=handle_file(pdf_file),
                claim=row["claim"],
                top_k=top_k,
                method=method,
                api_name="/retrieve_evidence",
            )
            evidences[index] = evidence
            end_time = time.time()
            retrieval_time[index] = round(end_time - start_time, 2)
            prompt_tokens[index] = _prompt_tokens
            completion_tokens[index] = _completion_tokens

        except:
            # If an error occured, print the file name and error message
            error_type, error_value, error_traceback = sys.exc_info()
            print(pdf_file)
            print(error_type)
            print(error_value)
            # Skip prediction step
            continue

        # Make the prediction
        result = client.predict(
            claim=row["claim"], evidence=evidence, api_name="/query_model"
        )
        label = result[1]["label"]
        labels[index] = label

# Create file name with method (without spaces) and top-k value (if not GPT)
if method == "GPT":
    output_file = f"predict_with_PDFs_{method}.csv"
else:
    output_file = f"predict_with_PDFs_{method.replace(" ", "")}_k{top_k}.csv"

# Convert results to DataFrame and save as CSV
df = pd.DataFrame(
    zip(labels, retrieval_time, prompt_tokens, completion_tokens, evidences),
    columns=[
        "label",
        "retrieval_time",
        "prompt_tokens",
        "completion_tokens",
        "evidence",
    ],
)
if not method == "GPT":
    df.drop(columns=["prompt_tokens", "completion_tokens"], inplace=True)
# For uploading test results, remove evidences retrieved from PDFs because they might contain copyrighted material
# Comment this line for local usage only:
# df.drop(columns = ["evidence"], inplace = True)
# Save results as CSV
df.to_csv(output_file, index=False)
