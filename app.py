import pandas as pd
import gradio as gr
from transformers import pipeline
import nltk
from retrieval_bm25s import retrieve_with_bm25s
from retrieval_bert import retrieve_with_deberta
from retrieval_gpt import retrieve_with_gpt
import os
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import spaces


def is_running_in_hf_spaces():
    """
    Detects if app is running in Hugging Face Spaces
    """
    return "SPACE_ID" in os.environ


if gr.NO_RELOAD:
    # Resource punkt_tab not found during application startup on HF spaces
    nltk.download("punkt_tab")

    # Keep track of the model name in a global variable so correct model is shown after page refresh
    # https://github.com/gradio-app/gradio/issues/3173
    MODEL_NAME = "jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint"
    pipe = pipeline(
        "text-classification",
        model=MODEL_NAME,
    )

    # Setup user feedback file for uploading to HF dataset
    # https://huggingface.co/spaces/Wauplin/space_to_dataset_saver
    # https://huggingface.co/docs/huggingface_hub/v0.16.3/en/guides/upload#scheduled-uploads
    USER_FEEDBACK_DIR = Path("user_feedback")
    USER_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    USER_FEEDBACK_PATH = USER_FEEDBACK_DIR / f"train-{uuid4()}.json"

    if is_running_in_hf_spaces():
        from huggingface_hub import CommitScheduler

        scheduler = CommitScheduler(
            repo_id="AI4citations-feedback",
            repo_type="dataset",
            folder_path=USER_FEEDBACK_DIR,
            path_in_repo="data",
        )


# Setup theme without background image
my_theme = gr.Theme.from_hub("NoCrypt/miku")
my_theme.set(body_background_fill="#FFFFFF", body_background_fill_dark="#000000")

# Define the HTML for Font Awesome
font_awesome_html = '<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">'

# Gradio interface setup
with gr.Blocks(theme=my_theme, head=font_awesome_html) as demo:

    # Layout
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                gr.Markdown("# AI4citations")
                gr.Markdown(
                    "## *AI-powered citation verification* ([more info](https://github.com/jedick/AI4citations))"
                )
            claim = gr.Textbox(
                label="Claim",
                info="aka hypothesis",
                placeholder="Input claim",
            )
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("Get Evidence from PDF"):
                        pdf_file = gr.File(
                            label="Upload PDF", type="filepath", height=120
                        )
                        with gr.Row():
                            retrieval_method = gr.Radio(
                                choices=["BM25S", "DeBERTa", "GPT"],
                                value="BM25S",
                                label="Retrieval Method",
                                info="Keyword search (BM25S) or AI (DeBERTa, GPT)",
                            )
                        get_evidence = gr.Button(value="Get Evidence and Submit")
                        top_k = gr.Slider(
                            1,
                            10,
                            value=5,
                            step=1,
                            label="Top k sentences",
                        )
                with gr.Column(scale=3):
                    evidence = gr.TextArea(
                        label="Evidence",
                        info="aka premise",
                        placeholder="Input evidence or use Get Evidence from PDF",
                    )
                    with gr.Row():
                        prompt_tokens = gr.Number(label="Prompt tokens", visible=False)
                        completion_tokens = gr.Number(
                            label="Completion tokens", visible=False
                        )
                    gr.Markdown(
                        """
                    ### App Usage:

                    - Input a **Claim**, then:
                        - Upload a PDF OR
                        - Input **Evidence** statements yourself
                    - To make the prediction with a PDF:
                        - Click **Get Evidence and Submit**
                    - To make the prediction after inputting or editing text:
                        - Hit 'Enter' in the **Claim** text box OR
                        - Hit 'Shift-Enter' in the **Evidence** text box
                    """
                    )
            with gr.Accordion("Sources", open=False):
                gr.Markdown(
                    """
                #### *Capstone project*
                - <i class="fa-brands fa-github"></i> [jedick/MLE-capstone-project](https://github.com/jedick/MLE-capstone-project) (project repo)
                - <i class="fa-brands fa-github"></i> [jedick/AI4citations](https://github.com/jedick/AI4citations) (app repo)
                #### *Text Classification*
                - <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" style="height: 1.2em; display: inline-block;"> [jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint](https://huggingface.co/jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint) (fine-tuned)
                - <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" style="height: 1.2em; display: inline-block;"> [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) (base)
                #### *Evidence Retrieval*
                - <i class="fa-brands fa-github"></i> [xhluca/bm25s](https://github.com/xhluca/bm25s) (BM25S)
                - <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" style="height: 1.2em; display: inline-block;"> [deepset/deberta-v3-large-squad2](https://huggingface.co/deepset/deberta-v3-large-squad2) (DeBERTa)
                - <img src="https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg" style="height: 1.2em; display: inline-block;"> [gpt-4o-mini-2024-07-18](https://platform.openai.com/docs/pricing) (GPT)
                #### *Datasets for fine-tuning*
                - <i class="fa-brands fa-github"></i> [allenai/SciFact](https://github.com/allenai/scifact) (SciFact)
                - <i class="fa-brands fa-github"></i> [ScienceNLP-Lab/Citation-Integrity](https://github.com/ScienceNLP-Lab/Citation-Integrity) (CitInt)
                #### *Other sources*
                - <img src="https://plos.org/wp-content/uploads/2020/01/logo-color-blue.svg" style="height: 1.4em; display: inline-block;"> [Medicine](https://doi.org/10.1371/journal.pmed.0030197), <i class="fa-brands fa-wikipedia-w"></i> [CRISPR](https://en.wikipedia.org/wiki/CRISPR) (evidence retrieval examples)
                - <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" style="height: 1.2em; display: inline-block;"> [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli/viewer/default/train?row=37&views%5B%5D=train) (MNLI example)
                - <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" style="height: 1.2em; display: inline-block;"> [NoCrypt/miku](https://huggingface.co/spaces/NoCrypt/miku) (theme)
                """
                )

        with gr.Column(scale=2):
            prediction = gr.Label(label="Prediction")
            with gr.Accordion("Feedback"):
                gr.Markdown(
                    "*Provide the correct label to help improve this app*<br>**NOTE:** The claim and evidence will also be saved"
                ),
                with gr.Row():
                    flag_support = gr.Button("Support")
                    flag_nei = gr.Button("NEI")
                    flag_refute = gr.Button("Refute")
                gr.Markdown(
                    "Feedback is uploaded every 5 minutes to [AI4citations-feedback](https://huggingface.co/datasets/jedick/AI4citations-feedback)"
                ),
            with gr.Accordion("Examples"):
                gr.Markdown("*Examples are run when clicked*"),
                with gr.Row():
                    support_example = gr.Examples(
                        examples="examples/Support",
                        label="Support",
                        inputs=[claim, evidence],
                        example_labels=pd.read_csv("examples/Support/log.csv")[
                            "label"
                        ].tolist(),
                    )
                    nei_example = gr.Examples(
                        examples="examples/NEI",
                        label="NEI",
                        inputs=[claim, evidence],
                        example_labels=pd.read_csv("examples/NEI/log.csv")[
                            "label"
                        ].tolist(),
                    )
                    refute_example = gr.Examples(
                        examples="examples/Refute",
                        label="Refute",
                        inputs=[claim, evidence],
                        example_labels=pd.read_csv("examples/Refute/log.csv")[
                            "label"
                        ].tolist(),
                    )
                retrieval_example = gr.Examples(
                    examples="examples/retrieval",
                    label="Get Evidence from PDF",
                    inputs=[pdf_file, claim],
                    example_labels=pd.read_csv("examples/retrieval/log.csv")[
                        "label"
                    ].tolist(),
                )
            # Create dropdown menu to select the model
            model = gr.Dropdown(
                choices=[
                    # TODO: For bert-base-uncased, how can we set num_labels = 2 in HF pipeline?
                    # (num_labels is available in AutoModelForSequenceClassification.from_pretrained)
                    # "bert-base-uncased",
                    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                    "jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint",
                ],
                value=MODEL_NAME,
                label="Model",
                info="Text classification model used for claim verification",
            )

    # Functions

    @spaces.GPU(duration=10)
    def query_model(claim, evidence):
        """
        Get prediction for a claim and evidence pair
        """
        prediction = {
            # Send a dictionary containing {"text", "text_pair"} keys; use top_k=3 to get results for all classes
            #   https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/pipelines#transformers.TextClassificationPipeline.__call__.inputs
            # Put evidence before claim
            #   https://github.com/jedick/MLE-capstone-project
            # Output {label: confidence} dictionary format as expected by gr.Label()
            #   https://github.com/gradio-app/gradio/issues/11170
            d["label"]: d["score"]
            for d in pipe({"text": evidence, "text_pair": claim}, top_k=3)
        }
        # Rename dictionary keys to use consistent labels across models
        prediction = {
            ("SUPPORT" if k in ["SUPPORT", "entailment"] else k): v
            for k, v in prediction.items()
        }
        prediction = {
            ("NEI" if k in ["NEI", "neutral"] else k): v for k, v in prediction.items()
        }
        prediction = {
            ("REFUTE" if k in ["REFUTE", "contradiction"] else k): v
            for k, v in prediction.items()
        }
        return prediction

    def select_model(model_name):
        """
        Select the specified model
        """
        global pipe, MODEL_NAME
        MODEL_NAME = model_name
        pipe = pipeline(
            "text-classification",
            model=MODEL_NAME,
        )

    # From gradio/client/python/gradio_client/utils.py
    def is_http_url_like(possible_url) -> bool:
        """
        Check if the given value is a string that looks like an HTTP(S) URL.
        """
        if not isinstance(possible_url, str):
            return False
        return possible_url.startswith(("http://", "https://"))

    def select_example(value, evt: gr.EventData):
        # Get the PDF file and claim from the event data
        claim, evidence = value[1]
        # Add the directory path
        return claim, evidence

    def select_retrieval_example(value, evt: gr.EventData):
        """
        Get the PDF file and claim from the event data.
        """
        pdf_file, claim = value[1]
        # Add the directory path
        if not is_http_url_like(pdf_file):
            pdf_file = f"examples/retrieval/{pdf_file}"
        return pdf_file, claim

    @spaces.GPU()
    def _retrieve_with_deberta(pdf_file, claim, top_k):
        """
        Retrieve evidence using DeBERTa
        """
        return retrieve_with_deberta(pdf_file, claim, top_k)

    def retrieve_evidence(pdf_file, claim, top_k, method):
        """
        Retrieve evidence using the selected method
        """
        if method == "BM25S":
            # Append 0 for number of prompt and completion tokens
            return retrieve_with_bm25s(pdf_file, claim, top_k), 0, 0
        elif method == "DeBERTa":
            return _retrieve_with_deberta(pdf_file, claim, top_k), 0, 0
        elif method == "GPT":
            return retrieve_with_gpt(pdf_file, claim)
        else:
            return f"Unknown retrieval method: {method}"

    def append_feedback(
        claim: str, evidence: str, model: str, prediction: str, user_label: str
    ) -> None:
        """
        Append input/outputs and user feedback to a JSON Lines file.
        """
        # Get the first label (prediction with highest probability)
        _prediction = next(iter(prediction))
        with USER_FEEDBACK_PATH.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "claim": claim,
                        "evidence": evidence,
                        "model": model,
                        "prediction": _prediction,
                        "user_label": user_label,
                        "datetime": datetime.now().isoformat(),
                    }
                )
            )
            f.write("\n")
        gr.Success(f"Saved your feedback: {user_label}", duration=2, title="Thank you!")

    def save_feedback_support(*args) -> None:
        """
        Save user feedback: Support
        """
        if is_running_in_hf_spaces():
            # Use a thread lock to avoid concurrent writes from different users.
            with scheduler.lock:
                append_feedback(*args, user_label="SUPPORT")
        else:
            append_feedback(*args, user_label="SUPPORT")

    def save_feedback_nei(*args) -> None:
        """
        Save user feedback: NEI
        """
        if is_running_in_hf_spaces():
            # Use a thread lock to avoid concurrent writes from different users.
            with scheduler.lock:
                append_feedback(*args, user_label="NEI")
        else:
            append_feedback(*args, user_label="NEI")

    def save_feedback_refute(*args) -> None:
        """
        Save user feedback: Refute
        """
        if is_running_in_hf_spaces():
            # Use a thread lock to avoid concurrent writes from different users.
            with scheduler.lock:
                append_feedback(*args, user_label="REFUTE")
        else:
            append_feedback(*args, user_label="REFUTE")

    def number_visible(value):
        """
        Show numbers (token counts) if GPT is selcted for retrieval
        """
        if value == "GPT":
            return gr.Number(visible=True)
        else:
            return gr.Number(visible=False)

    def slider_visible(value):
        """
        Hide slider (top_k) if GPT is selcted for retrieval
        """
        if value == "GPT":
            return gr.Slider(visible=False)
        else:
            return gr.Slider(visible=True)

    # Event listeners

    # Press Enter or Shift-Enter to submit
    gr.on(
        triggers=[claim.submit, evidence.submit],
        fn=query_model,
        inputs=[claim, evidence],
        outputs=prediction,
    )

    # Get evidence from PDF and run the model
    gr.on(
        triggers=[get_evidence.click],
        fn=retrieve_evidence,
        inputs=[pdf_file, claim, top_k, retrieval_method],
        outputs=[evidence, prompt_tokens, completion_tokens],
    ).then(
        fn=query_model,
        inputs=[claim, evidence],
        outputs=prediction,
        api_name=False,
    )

    # Handle "Support" examples
    gr.on(
        triggers=[support_example.dataset.select],
        fn=select_example,
        inputs=support_example.dataset,
        outputs=[claim, evidence],
        api_name=False,
    ).then(
        fn=query_model,
        inputs=[claim, evidence],
        outputs=prediction,
        api_name=False,
    )

    # Handle "NEI" examples
    gr.on(
        triggers=[nei_example.dataset.select],
        fn=select_example,
        inputs=nei_example.dataset,
        outputs=[claim, evidence],
        api_name=False,
    ).then(
        fn=query_model,
        inputs=[claim, evidence],
        outputs=prediction,
        api_name=False,
    )

    # Handle "Refute" examples
    gr.on(
        triggers=[refute_example.dataset.select],
        fn=select_example,
        inputs=refute_example.dataset,
        outputs=[claim, evidence],
        api_name=False,
    ).then(
        fn=query_model,
        inputs=[claim, evidence],
        outputs=prediction,
        api_name=False,
    )

    # Handle evidence retrieval examples: get evidence from PDF and run the model
    gr.on(
        triggers=[retrieval_example.dataset.select],
        fn=select_retrieval_example,
        inputs=retrieval_example.dataset,
        outputs=[pdf_file, claim],
        api_name=False,
    ).then(
        fn=retrieve_evidence,
        inputs=[pdf_file, claim, top_k, retrieval_method],
        outputs=[evidence, prompt_tokens, completion_tokens],
        api_name=False,
    ).then(
        fn=query_model,
        inputs=[claim, evidence],
        outputs=prediction,
        api_name=False,
    )

    # Change the model then update the predictions
    model.change(
        fn=select_model,
        inputs=model,
    ).then(
        fn=query_model,
        inputs=[claim, evidence],
        outputs=prediction,
        api_name=False,
    )

    # Log user feedback when button is clicked
    flag_support.click(
        fn=save_feedback_support,
        inputs=[claim, evidence, model, prediction],
        outputs=None,
        api_name=False,
    )
    flag_nei.click(
        fn=save_feedback_nei,
        inputs=[claim, evidence, model, prediction],
        outputs=None,
        api_name=False,
    )
    flag_refute.click(
        fn=save_feedback_refute,
        inputs=[claim, evidence, model, prediction],
        outputs=None,
        api_name=False,
    )

    # Change visibility of top-k slider and token counts if GPT is selected for retrieval
    retrieval_method.change(
        number_visible, retrieval_method, prompt_tokens, api_name=False
    )
    retrieval_method.change(
        number_visible, retrieval_method, completion_tokens, api_name=False
    )
    retrieval_method.change(slider_visible, retrieval_method, top_k, api_name=False)


if __name__ == "__main__":
    # allowed_paths is needed to upload PDFs from specific example directory
    demo.launch(allowed_paths=[f"{os.getcwd()}/examples/retrieval"])
