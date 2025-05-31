import re
import fitz  # pip install pymupdf
from unidecode import unidecode
from nltk.tokenize import sent_tokenize
import bm25s


def retrieve_with_bm25s(pdf_file, claim, top_k=10):

    # Get PDF file as binary
    with open(pdf_file, mode="rb") as f:
        pdf_file_bytes = f.read()

    # Extract text from the PDF
    pdf_doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
    pdf_text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        pdf_text += page.get_text("text")

    # Clean text
    # pdf_text = 'In §3.1, we ﬁnd\nthat dis-\ntractor abstracts.')
    # clean_text = 'In SS3.1, we find that distractor abstracts.'
    # Remove hyphens at end of lines
    clean_text = re.sub("-\n", "", pdf_text)
    # Replace remaining newline characters with space
    clean_text = re.sub("\n", " ", clean_text)
    # Replace unicode with ascii
    clean_text = unidecode(clean_text)

    # Parse text into sentences to build the corpus
    corpus = sent_tokenize(clean_text)
    # Tokenize the corpus
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
    # Initialize the BM25 model
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    # Tokenize the claim
    query_tokens = bm25s.tokenize(claim)

    # Get top k results
    # Use int(k) in case we get str value (as in retrieval example)
    results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=int(top_k))
    ## Print results
    # for i in range(results.shape[1]):
    #    doc, score = results[0, i], scores[0, i]
    #    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

    # Join sentences and return results
    results = " ".join(results[0])
    return results
