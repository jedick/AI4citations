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

# Custom CSS to center content
custom_css = """
.center-content {
    text-align: center;
    display:block;
}
"""

# Define the HTML for Font Awesome
font_awesome_html = '<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">'

# Callback for user feedback
callback = gr.CSVLogger()

# Gradio interface setup
with gr.Blocks(theme=my_theme, css=custom_css, head=font_awesome_html) as demo:

    # Layout
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                gr.Markdown("# AI4citations")
                gr.Markdown("## *AI-powered scientific citation verification*")
            claim = gr.Textbox(
                label="1. Claim",
                info="aka hypothesis",
                placeholder="Input claim",
            )
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("Get Evidence from PDF"):
                        pdf_file = gr.File(
                            label="Upload PDF", type="filepath", height=120
                        )
                        get_evidence = gr.Button(value="Get Evidence")
                        top_k = gr.Slider(
                            1,
                            10,
                            value=5,
                            step=1,
                            interactive=True,
                            label="Top k sentences",
                        )
                with gr.Column(scale=3):
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
            label = gr.Label(label="Results")
            with gr.Accordion("Feedback"):
                gr.Markdown(
                    "*Click a button with the correct label to improve this app*<br>**NOTE:** the claim and evidence will also be logged"
                ),
                with gr.Row():
                    flag_support = gr.Button("Support")
                    flag_nei = gr.Button("NEI")
                    flag_refute = gr.Button("Refute")
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

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                    ### Usage:

                    1. Input a **Claim**
                    2. Input **Evidence** statements OR upload a PDF and click **Get Evidence**
                    """
                    )
                with gr.Column(scale=2):
                    gr.Markdown(
                        """
                    ### To make predictions:

                    - Hit 'Enter' in the **Claim** text box,
                    - Hit 'Shift-Enter' in the **Evidence** text box, or
                    - Click **Get Evidence**
                    """
                    )

        with gr.Column(scale=2):
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
            with gr.Accordion("Sources", open=False, elem_classes=["center_content"]):
                gr.Markdown(
                    """
                #### *Capstone project*
                - <i class="fa-brands fa-github"></i> [jedick/MLE-capstone-project](https://github.com/jedick/MLE-capstone-project) (project repo)
                - <i class="fa-brands fa-github"></i> [jedick/AI4citations](https://github.com/jedick/AI4citations) (app repo)
                """
                )
                gr.Markdown(
                    """
                #### *Models*
                - <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" style="height: 1.2em; display: inline-block;"> [jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint](https://huggingface.co/jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint) (fine-tuned)
                - <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" style="height: 1.2em; display: inline-block;"> [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) (base)
                """
                )
                gr.Markdown(
                    """
                #### *Datasets for fine-tuning*
                - <i class="fa-brands fa-github"></i> [allenai/SciFact](https://github.com/allenai/scifact) (SciFact)
                - <i class="fa-brands fa-github"></i> [ScienceNLP-Lab/Citation-Integrity](https://github.com/ScienceNLP-Lab/Citation-Integrity) (CitInt)
                """
                )
                gr.Markdown(
                    """
                #### *Other sources*
                - <i class="fa-brands fa-github"></i> [xhluca/bm25s](https://github.com/xhluca/bm25s) (evidence retrieval)
                - <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" style="height: 1.2em; display: inline-block;"> [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli/viewer/default/train?row=37&views%5B%5D=train) (MNLI example)
                - <img src="https://plos.org/wp-content/uploads/2020/01/logo-color-blue.svg" style="height: 1.4em; display: inline-block;"> [Medicine](https://doi.org/10.1371/journal.pmed.0030197), <i class="fa-brands fa-wikipedia-w"></i> [CRISPR](https://en.wikipedia.org/wiki/CRISPR) (get evidence examples)
                - <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" style="height: 1.2em; display: inline-block;"> [NoCrypt/miku](https://huggingface.co/spaces/NoCrypt/miku) (theme)
                """
                )

    # Setup callback to log user feedback
    callback.setup([claim, evidence, label], "user_feedback")

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
        fn=select_model,
        inputs=dropdown,
    ).then(
        fn=query_model,
        inputs=[claim, evidence],
        outputs=[prediction, label],
        api_name=False,
    )

    # Log user feedback when button is clicked
    flag_support.click(
        lambda *args: callback.flag(list(args), flag_option="Support"),
        [claim, evidence, label],
        None,
        preprocess=False,
    )
    flag_nei.click(
        lambda *args: callback.flag(list(args), flag_option="NEI"),
        [claim, evidence, label],
        None,
        preprocess=False,
    )
    flag_refute.click(
        lambda *args: callback.flag(list(args), flag_option="Refute"),
        [claim, evidence, label],
        None,
        preprocess=False,
    )


if __name__ == "__main__":
    # allowed_paths is needed to upload PDFs from specific example directory
    demo.launch(allowed_paths=[f"{os.getcwd()}/examples/retrieval"])
