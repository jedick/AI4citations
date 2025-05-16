import pandas as pd
import gradio as gr
from retrieval import retrieve_from_pdf

if gr.NO_RELOAD:
    from transformers import pipeline

    # Keep track of the model name in a global variable so correct model is shown after page refresh
    # https://github.com/gradio-app/gradio/issues/3173
    MODEL_NAME = "jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint"
    pipe = pipeline(
        "text-classification",
        model=MODEL_NAME,
    )


def query_model(claim, evidence):
    """
    Get prediction for a pair of claim and evidence
    """
    prediction = {
        # Send a dictionary containing {"text", "text_pair"} keys; use top_k=3 to get results for all classes
        #   https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/pipelines#transformers.TextClassificationPipeline.__call__.inputs
        # Put evidence before claim
        #   https://github.com/jedick/ML-capstone-project
        # Output {label: confidence} dictionary format as expected by gr.Label()
        #   https://github.com/gradio-app/gradio/issues/11170
        d["label"]: d["score"]
        for d in pipe({"text": evidence, "text_pair": claim}, top_k=3)
    }
    # Return two instances of the prediction to send to different Gradio components
    return prediction, prediction


def query_model_for_examples(claim, evidence):
    """
    A duplicate of the previous function, used to keep the API names clean
    """
    prediction = {
        d["label"]: d["score"]
        for d in pipe({"text": evidence, "text_pair": claim}, top_k=3)
    }
    return prediction, prediction


# Function to select the model
def select_model(model_name):
    global pipe, MODEL_NAME
    MODEL_NAME = model_name
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


def change_visualization(choice):
    if choice == "barplot":
        barplot = gr.update(visible=True)
        label = gr.update(visible=False)
    elif choice == "label":
        barplot = gr.update(visible=False)
        label = gr.update(visible=True)
    return barplot, label


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
                ## Scientific citation verification

                *Press Enter in a textbox or click Submit to run the model.*
                """
                )
                gr.Markdown(
                    """
                ### Three ways to use this app

                1. **Claim verification**: Input a claim and evidence
                2. **Evidence retrieval**: Input a claim to get evidence from PDF
                3. **Claim extraction**: Input a text to get claim from text
                """
                )
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
            claim = gr.Textbox(
                label="Claim",
                info="aka hypothesis",
                placeholder="Input claim or use Get Claim from Text",
            )
            evidence = gr.TextArea(
                label="Evidence",
                info="aka premise",
                placeholder="Input evidence or use Get Evidence from PDF",
            )
            with gr.Row():
                with gr.Accordion("Get Claim from Text", open=False):
                    text = gr.TextArea(
                        label="Text",
                        placeholder="Under construction!",
                        interactive=False,
                    )
                with gr.Accordion("Get Evidence from PDF", open=False):
                    pdf_file = gr.File(label="Upload PDF", type="filepath")
                    get_evidence = gr.Button(value="Get Evidence")
                    top_k = gr.Slider(
                        1,
                        10,
                        value=5,
                        step=1,
                        interactive=True,
                        label="Top k sentences",
                    )
            submit = gr.Button("Submit")

        with gr.Column(scale=2):
            radio = gr.Radio(["barplot", "label"], value="barplot", label="Results")
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
            )
            label = gr.Label(visible=False)
            with gr.Accordion("Examples", open=False):
                gr.Markdown(
                    "*Prediction performance with jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint:*"
                ),
                gr.Examples(
                    examples="examples/accurate",
                    inputs=[claim, evidence],
                    outputs=[prediction, label],
                    fn=query_model_for_examples,
                    label="Accurate",
                    run_on_click=True,
                    example_labels=pd.read_csv("examples/accurate/log.csv")[
                        "label"
                    ].tolist(),
                )
                gr.Examples(
                    examples="examples/inaccurate",
                    inputs=[claim, evidence],
                    outputs=[prediction, label],
                    fn=query_model_for_examples,
                    label="Inaccurate",
                    run_on_click=True,
                    example_labels=pd.read_csv("examples/inaccurate/log.csv")[
                        "label"
                    ].tolist(),
                )
            gr.Markdown(
                """
            ### Sources
            - ML project: [jedick/ML-capstone-project](https://github.com/jedick/ML-capstone-project)
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

    # Event listeners

    # Click the submit button or press Enter to submit
    gr.on(
        triggers=[claim.submit, evidence.submit, submit.click],
        fn=query_model,
        inputs=[claim, evidence],
        outputs=[prediction, label],
    )

    # Clear the previous predictions when the model is changed
    gr.on(
        triggers=[dropdown.select],
        fn=lambda: "Model changed! Waiting for updated predictions...",
        outputs=[prediction],
        api_name=False,
    )

    # Update the predictions after changing the model
    dropdown.change(
        fn=select_model,
        inputs=dropdown,
    ).then(
        fn=query_model,
        inputs=[claim, evidence],
        outputs=[prediction, label],
        api_name=False,
    )

    # Get evidence from PDF
    gr.on(
        triggers=[pdf_file.upload, get_evidence.click],
        fn=retrieve_from_pdf,
        inputs=[pdf_file, claim, top_k],
        outputs=evidence,
    )

    # Change visualization
    radio.change(
        fn=change_visualization,
        inputs=radio,
        outputs=[barplot, label],
        api_name=False,
    )

demo.launch()
