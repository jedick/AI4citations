[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/jedick/AI4citations)
[![Codecov test coverage](https://codecov.io/gh/jedick/AI4citations/graph/badge.svg)](https://app.codecov.io/gh/jedick/AI4citations)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# AI-Powered Citation Verification

The integrity of scientific literature depends on citations that are supported by the referenced source material.
These citations are sometimes inaccurate, contributing to unverified claims.
Automatic detection of citation accuracy can help writers and editors improve the overall quality of scientific literature.

AI4ciations is an easy-to-use app for citation verification that leverages a claim verification model trained on domain-specific datasets.
For details on the data preprocessing, model fine-tuning, and baseline comparisons, see the repo for this [ML engineering capstone project](https://github.com/jedick/MLE-capstone-project).

Features:

- **Claim verification**: Input a pair of claim and evidence statements and predict a label
  - Labels are Support, Refute, or Not Enough Information (NEI)
- Model selection: Choose from a fine-tuned model (default) or the pretrained base model
  - The [default model](https://huggingface.co/jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint) was fine-tuned on two datasets, [SciFact](https://github.com/allenai/scifact) and [Citation-Integrity](https://github.com/ScienceNLP-Lab/Citation-Integrity/)
  - The [base model](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) is DeBERTa pre-trained on multiple natural language inference (NLI) datasets
  - See this [blog post](https://jedick.github.io/blog/experimenting-with-transformer-models-for-citation-verification/) for more information on fine-tuning
- **Evidence retrieval**: Get evidence from a PDF related to the given claim (see details below)

## Running the app

- Install the requirements with the `pip install -r requirements.txt`
- Run `gradio app.py` to launch the app
- Browse to the generated URL (e.g. `http://127.0.0.1:7860`)

Screenshot of app with [example text](https://huggingface.co/datasets/nyu-mll/multi_nli/viewer/default/train?row=37&views%5B%5D=train):

![Screenshot of AI4citations app](./images/AI4citations_screenshot.png)

## Retrieval methods

First, some definitions:

- *Gold evidence* is the abstract from the cited paper used by human annotators to label each claim
- *Retrieved evidence* is sentences retrieved from the PDF of the cited paper
  - The claim was used as the query
  - Sentences were retrieved only from the cited PDF
  - The number of retrieved sentences (top k) was set to 5 or 10

Retrieval methods implemented in the app:

- **BM25**: Keyword-based retrieval using BM25-Sparse ranking algorithm ([BM25S](https://github.com/xhluca/bm25s))
- **BERT**: AI-based retrieval using pretrained DeBERTa language model ([deepset/deberta-v3-large-squad2](https://huggingface.co/deepset/deberta-v3-large-squad2))
- **GPT**: AI-based retrieval using OpenAI's GPT ([gpt-4o-mini-2024-07-18](https://platform.openai.com/docs/pricing))

### Evaluation

Predictions were made on the SciFact test set, with the following results for the claim verification task:

| Retrieval method | Macro F1 (k=5) | Macro F1 (k=10) | Avg. retrieval time (s) |
| - | - | - | - |
| BM25 | 0.649 | 0.654 | 0.36 |
| BERT | 0.610 | 0.591 | 7.00 |

For comparison, Macro F1 for claim verification using gold evidence (abstracts) is 0.834.

## Acknowledgments

- App built with [Gradio](https://github.com/gradio-app/gradio)
- BERT retrieval code writen with AI assistance (Claude Sonnet 4)
