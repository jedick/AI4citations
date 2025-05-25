import pandas as pd
import gradio as gr
from transformers import pipeline
import nltk
from retrieval import retrieve_from_pdf
import os

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


def prediction_to_df(prediction=None):
    """
    Convert prediction text to DataFrame for barplot
    """
    if prediction is None or prediction == "":
        # Show an empty plot for app initialization or auto-reload
        prediction = {"SUPPORT": 0, "NEI": 0, "REFUTE": 0}
    elif "Model" in prediction:
        # Show full-height bars when the model is changed
        prediction = {"SUPPORT": 1, "NEI": 1, "REFUTE": 1}
    else:
        # Convert predictions text to dictionary
        prediction = eval(prediction)
        # Rename dictionary keys to use consistent labels across models
        prediction = {
            ("SUPPORT" if k == "entailment" else k): v for k, v in prediction.items()
        }
        prediction = {
            ("NEI" if k == "neutral" else k): v for k, v in prediction.items()
        }
        prediction = {
            ("REFUTE" if k == "contradiction" else k): v for k, v in prediction.items()
        }
        # Use custom order for labels (pipe() returns labels in descending order of softmax score)
        labels = ["SUPPORT", "NEI", "REFUTE"]
        prediction = {k: prediction[k] for k in labels}
    # Convert dictionary to DataFrame with one column (Probability)
    df = pd.DataFrame.from_dict(prediction, orient="index", columns=["Probability"])
    # Move the index to the Class column
    return df.reset_index(names="Class")


# Setup theme without background image
my_theme = gr.Theme.from_hub("NoCrypt/miku")
my_theme.set(body_background_fill="#FFFFFF", body_background_fill_dark="#000000")

# Gradio interface setup
with gr.Blocks(theme=my_theme) as demo:

    # Layout
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                gr.Markdown(
                    """
                # AI4citations

                ### Usage:

                1. Input a **Claim**
                2. Input **Evidence** statements
                - *Optional:* Upload a PDF and click Get Evidence
                """
                )
                gr.Markdown(
                    """
                ## *AI-powered citation verification*

                ### To make predictions:

                - Hit 'Enter' in the **Claim** text box,
                - Hit 'Shift-Enter' in the **Evidence** text box, or
                - Click Get Evidence
                """
                )
            claim = gr.Textbox(
                label="1. Claim",
                info="aka hypothesis",
                placeholder="Input claim or use Get Claim from Text",
            )
            with gr.Row():
                with gr.Accordion("Get Evidence from PDF", open=True):
                    pdf_file = gr.File(label="Upload PDF", type="filepath", height=120)
                    get_evidence = gr.Button(value="Get Evidence")
                    top_k = gr.Slider(
                        1,
                        10,
                        value=5,
                        step=1,
                        interactive=True,
                        label="Top k sentences",
                    )
                evidence = gr.TextArea(
                    label="2. Evidence",
                    info="aka premise",
                    placeholder="Input evidence or use Get Evidence from PDF",
                )
            submit = gr.Button("3. Submit", visible=False)

        with gr.Column(scale=2):
            # Keep the prediction textbox hidden
            with gr.Accordion(visible=False):
                prediction = gr.Textbox(label="Prediction")
            barplot = gr.BarPlot(
                prediction_to_df,
                x="Class",
                y="Probability",
                color="Class",
                color_map={"SUPPORT": "green", "NEI": "#888888", "REFUTE": "#FF8888"},
                inputs=prediction,
                y_lim=([0, 1]),
                visible=False,
            )
            label = gr.Label()
            with gr.Accordion("Settings", open=False):
                # Create dropdown menu to select the model
                dropdown = gr.Dropdown(
                    choices=[
                        # TODO: For bert-base-uncased, how can we set num_labels = 2 in HF pipeline?
                        # (num_labels is available in AutoModelForSequenceClassification.from_pretrained)
                        # "bert-base-uncased",
                        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                        "jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint",
                    ],
                    value=MODEL_NAME,
                    label="Model",
                )
                radio = gr.Radio(["label", "barplot"], value="label", label="Results")
            with gr.Accordion("Examples", open=False):
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
                    label="Retrieval",
                    inputs=[pdf_file, claim],
                    example_labels=pd.read_csv("examples/retrieval/log.csv")[
                        "label"
                    ].tolist(),
                )
            gr.Markdown(
                """
            ### Sources
            - ML engineering project: [jedick/MLE-capstone-project](https://github.com/jedick/MLE-capstone-project)
                - App repository: [jedick/AI4citations](https://github.com/jedick/AI4citations)
                - Fine-tuned model: [jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint](https://huggingface.co/jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint)
            - Datasets used for fine-tuning
                - SciFact: [allenai/SciFact](https://github.com/allenai/scifact)
                - Citation-Integrity (CitInt): [ScienceNLP-Lab/Citation-Integrity](https://github.com/ScienceNLP-Lab/Citation-Integrity)
            - Base model: [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
            - Evidence retrieval: [xhluca/bm25s](https://github.com/xhluca/bm25s)
            - Gradio theme: [NoCrypt/miku](https://huggingface.co/spaces/NoCrypt/miku)
            """
            )

    # Functions

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
        # Return two instances of the prediction to send to different Gradio components
        return prediction, prediction

    def use_model(model_name):
        """
        Use the specified model
        """
        global pipe, MODEL_NAME
        MODEL_NAME = model_name
        pipe = pipeline(
            "text-classification",
            model=MODEL_NAME,
        )

    def change_visualization(choice):
        if choice == "barplot":
            barplot = gr.update(visible=True)
            label = gr.update(visible=False)
        elif choice == "label":
            barplot = gr.update(visible=False)
            label = gr.update(visible=True)
        return barplot, label

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

    # Event listeners

    # Click the submit button or press Enter to submit
    gr.on(
        triggers=[claim.submit, evidence.submit, submit.click],
        fn=query_model,
        inputs=[claim, evidence],
        outputs=[prediction, label],
    )

    # Get evidence from PDF and run the model
    gr.on(
        triggers=[get_evidence.click],
        fn=retrieve_from_pdf,
        inputs=[pdf_file, claim, top_k],
        outputs=evidence,
    ).then(
        fn=query_model,
        inputs=[claim, evidence],
        outputs=[prediction, label],
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
        outputs=[prediction, label],
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
        outputs=[prediction, label],
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
        outputs=[prediction, label],
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
        fn=retrieve_from_pdf,
        inputs=[pdf_file, claim, top_k],
        outputs=evidence,
        api_name=False,
    ).then(
        fn=query_model,
        inputs=[claim, evidence],
        outputs=[prediction, label],
        api_name=False,
    )

    # Change visualization
    radio.change(
        fn=change_visualization,
        inputs=radio,
        outputs=[barplot, label],
        api_name=False,
    )

    # Clear the previous predictions when the model is changed
    gr.on(
        triggers=[dropdown.select],
        fn=lambda: "Model changed! Waiting for updated predictions...",
        outputs=[prediction],
        api_name=False,
    )

    # Change the model the update the predictions
    dropdown.change(
        fn=use_model,
        inputs=dropdown,
    ).then(
        fn=query_model,
        inputs=[claim, evidence],
        outputs=[prediction, label],
        api_name=False,
    )


if __name__ == "__main__":
    # allowed_paths is needed to upload PDFs from specific example directory
    demo.launch(allowed_paths=[f"{os.getcwd()}/examples/retrieval"])
