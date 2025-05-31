import re
import fitz  # pip install pymupdf
from unidecode import unidecode
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer
import torch
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTRetriever:
    """
    BERT-based evidence retrieval using extractive question answering
    """

    def __init__(self, model_name: str = "deepset/deberta-v3-large-squad2"):
        """
        Initialize the BERT evidence retriever

        Args:
            model_name: HuggingFace model for question answering
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )
        # Maximum context length for the model
        self.max_length = self.tokenizer.model_max_length
        logger.info(f"Initialized BERT retriever with model: {model_name}")

    def _extract_and_clean_text(self, pdf_file: str) -> str:
        """
        Extract and clean text from PDF file

        Args:
            pdf_file: Path to PDF file

        Returns:
            Cleaned text from PDF
        """
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
        # Remove hyphens at end of lines
        clean_text = re.sub("-\n", "", pdf_text)
        # Replace remaining newline characters with space
        clean_text = re.sub("\n", " ", clean_text)
        # Replace unicode with ascii
        clean_text = unidecode(clean_text)

        return clean_text

    def _chunk_text(self, text: str, max_chunk_size: int = 3000) -> List[str]:
        """
        Split text into chunks that fit within model context window

        Args:
            text: Input text to chunk
            max_chunk_size: Maximum size per chunk

        Returns:
            List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _format_claim_as_question(self, claim: str) -> str:
        """
        Convert a claim into a question format for better QA performance

        Args:
            claim: Input claim

        Returns:
            Question formatted for QA model
        """
        # Simple heuristics to convert claims to questions
        claim = claim.strip()

        # If already a question, return as is
        if claim.endswith("?"):
            return claim

        # Convert common claim patterns to questions
        if claim.lower().startswith(("the ", "a ", "an ")):
            return f"What evidence supports that {claim.lower()}?"
        elif "is" in claim.lower() or "are" in claim.lower():
            return f"Is it true that {claim.lower()}?"
        elif "can" in claim.lower() or "could" in claim.lower():
            return f"{claim}?"
        else:
            return f"What evidence supports the claim that {claim.lower()}?"

    def retrieve_evidence(self, pdf_file: str, claim: str, top_k: int = 5) -> str:
        """
        Retrieve evidence from PDF using BERT-based question answering

        Args:
            pdf_file: Path to PDF file
            claim: Claim to find evidence for
            k: Number of evidence passages to retrieve

        Returns:
            Combined evidence text
        """
        try:
            # Extract and clean text from PDF
            clean_text = self._extract_and_clean_text(pdf_file)

            # Convert claim to question format
            question = self._format_claim_as_question(claim)

            # Split text into manageable chunks
            chunks = self._chunk_text(clean_text)

            # Get answers from each chunk
            answers = []
            for i, chunk in enumerate(chunks):
                try:
                    result = self.qa_pipeline(
                        question=question, context=chunk, max_answer_len=200, top_k=1
                    )

                    # Handle both single answer and list of answers
                    if isinstance(result, list):
                        result = result[0]

                    if result["score"] > 0.1:  # Confidence threshold
                        # Extract surrounding context for better evidence
                        answer_text = result["answer"]
                        start_idx = max(0, chunk.find(answer_text) - 100)
                        end_idx = min(
                            len(chunk), chunk.find(answer_text) + len(answer_text) + 100
                        )
                        context = chunk[start_idx:end_idx].strip()

                        answers.append(
                            {"text": context, "score": result["score"], "chunk_idx": i}
                        )

                except Exception as e:
                    logger.warning(f"Error processing chunk {i}: {str(e)}")
                    continue

            # Sort by confidence score and take top k
            answers.sort(key=lambda x: x["score"], reverse=True)
            top_answers = answers[:top_k]

            # Combine evidence passages
            if top_answers:
                evidence_texts = [answer["text"] for answer in top_answers]
                combined_evidence = " ".join(evidence_texts)
                return combined_evidence
            else:
                logger.warning("No evidence found with sufficient confidence")
                return "No relevant evidence found in the document."

        except Exception as e:
            logger.error(f"Error in BERT evidence retrieval: {str(e)}")
            return f"Error retrieving evidence: {str(e)}"


def retrieve_with_deberta(pdf_file: str, claim: str, top_k: int = 5) -> str:
    """
    Wrapper function for DeBERTa-based evidence retrieval
    Compatible with the existing BM25S interface

    Args:
        pdf_file: Path to PDF file
        claim: Claim to find evidence for
        top_k: Number of evidence passages to retrieve

    Returns:
        Retrieved evidence text
    """
    # Initialize retriever (in production, this should be cached)
    retriever = BERTRetriever()
    return retriever.retrieve_evidence(pdf_file, claim, top_k)


# Alternative lightweight model for faster inference
class DistilBERTRetriever(BERTRetriever):
    """
    Lightweight version using smaller, faster models
    """

    def __init__(self):
        super().__init__(model_name="distilbert-base-cased-distilled-squad")


def retrieve_with_distilbert(pdf_file: str, claim: str, top_k: int = 5) -> str:
    """
    Fast DistilBERT-based evidence retrieval

    Args:
        pdf_file: Path to PDF file
        claim: Claim to find evidence for
        top_k: Number of evidence passages to retrieve

    Returns:
        Retrieved evidence text
    """
    retriever = DistilBERTRetriever()
    return retriever.retrieve_evidence(pdf_file, claim, top_k)
