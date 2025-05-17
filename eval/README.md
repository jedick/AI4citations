## AI4citations/evals

Scripts and data to evaluate the app on the SciFact test set.

- `scifact_test_data.R`: Script to generate the following CSV from `*.jsonl` claim and corpus files [available here](https://github.com/jedick/ML-capstone-project/tree/main/data/scifact)
- `scifact_test_data.csv`: The test dataset including claim, label, corpus_id, title, and abstract
- `scifact_test_data_sources.csv`: For each unique corpus_id, the Semantic Scholar URL and DOI for locating PDFs
  - *Because of copyright, PDFs are not uploaded here*
- `predict_with_abstracts.py`: Script to make predictions with the gold evidence (abstracts)
- `predict_with_PDFs.py`: Script to make predictions with evidence sentences retrieved from the PDF
- `predict_with_abstracts.csv`, `predict_with_PDFs_k5.csv`, `predict_with_PDFs_k10.csv`: Predicted labels (file names include top k sentences for retrieval)
- `calculate_F1_scores.py`: Script to calculate F1 scores from the true and predicted labels
