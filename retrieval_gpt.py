from openai import OpenAI
import os
from typing import Tuple


def retrieve_with_gpt(pdf_file: str, claim: str) -> Tuple[str, int, int]:
    """
    Retrieve evidence from PDF using GPT

    Args:
        pdf_file: Path to PDF file
        claim: Claim to find evidence for

    Returns:
        Tuple with retrieved evidence text, prompt tokens, and completion tokens
    """

    model = "gpt-4o-mini-2024-07-18"

    prompt = """Retrieve sentences from the PDF (title, abstract, text, sections, not References/Bibliography) to support or refute this claim. \
    Summarize any information from images. \
    Respond only with verbatim sentences from the text and/or summarized sentences from images. \
    If no conclusive evidence is found, respond with the five sentences that are most relevant to the claim. \
    Combine all sentences into one response without quotation marks or line numbers. \
    """

    prompt = "".join([prompt, f"CLAIM: {claim}"])

    client = OpenAI()

    file = client.files.create(file=open(pdf_file, "rb"), purpose="user_data")

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_id": file.id,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    return (
        completion.choices[0].message.content,
        completion.usage.prompt_tokens,
        completion.usage.completion_tokens,
    )
