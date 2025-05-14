import gradio as gr
import pandas as pd

if gr.NO_RELOAD:
    from transformers import pipeline

    # Keep track of the model name in a global variable so correct model is shown after page refresh
    # https://github.com/gradio-app/gradio/issues/3173
    MODEL_NAME = "jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint"
    pipe = pipeline(
        "text-classification",
        model=MODEL_NAME,
    )


def predict(claim, evidence):
    # Send a dictionary containing {"text", "text_pair"} keys; use top_k=3 to get results for all classes
    #   https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/pipelines#transformers.TextClassificationPipeline.__call__.inputs
    # Put evidence before claim
    #   https://github.com/jedick/ML-capstone-project
    # Output {label: confidence} dictionary format as expected by gr.Label()
    #   https://github.com/gradio-app/gradio/issues/11170
    prediction = {
        d["label"]: d["score"]
        for d in pipe({"text": evidence, "text_pair": claim}, top_k=3)
    }
    # Return two instances of the prediction to send to different Gradio components
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
    if prediction is None:
        # This shows a half-filled plot for app auto-reload
        # (running with gradio app.py, not python app.py)
        prediction = {"SUPPORT": 0.5, "NEI": 0.5, "REFUTE": 0.5}
    elif prediction == "":
        # This shows an empty plot for app initialization
        prediction = {"SUPPORT": 0, "NEI": 0, "REFUTE": 0}
    elif "Model" in prediction:
        # This shows full-height bars when the model is changed
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


# Gradio interface setup
with gr.Blocks() as demo:

    # Layout
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                """
            # AI4citations
            ### Scientific citation verification

            - Basic claim verification: Enter a claim and evidence
            - Evidence retrieval: Enter a claim then get evidence from a PDF
            - Claim extraction: Enter text, get claim from text, then get evidence from a PDF
            """
            )
            # Create dropdown menu to select the model
            dropdown = gr.Dropdown(
                choices=[
                    # TODO: For bert-base-uncased, how can we set num_labels = 2 in HF pipeline (like we could in AutoModelForSequenceClassification.from_pretrained)?
                    # "bert-base-uncased",
                    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                    "jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint",
                ],
                value=MODEL_NAME,
                label="Select Model",
            )
            claim = gr.Textbox(label="Enter the claim (or hypothesis)")
            evidence = gr.Textbox(label="Enter the evidence (or premise)")
            submit = gr.Button("Submit")

        with gr.Column(scale=2):
            radio = gr.Radio(
                ["barplot", "label"], value="barplot", label="Visualization"
            )
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
            gr.Examples(
                examples="ex_accurate",
                inputs=[claim, evidence],
                outputs=[prediction, label],
                fn=predict,
                label="Examples of accurate predictions",
                run_on_click=True,
                example_labels=pd.read_csv("ex_accurate/log.csv")["label"].tolist(),
            )
            gr.Examples(
                examples="ex_inaccurate",
                inputs=[claim, evidence],
                outputs=[prediction, label],
                fn=predict,
                label="Examples of inaccurate predictions",
                run_on_click=True,
                example_labels=pd.read_csv("ex_inaccurate/log.csv")["label"].tolist(),
            )
            gr.Markdown(
                """
            ### About
            - App repo: <https://github.com/jedick/AI4citations>
            - ML project: <https://github.com/jedick/ML-capstone-project>
            - Base model: <https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli>
            - Fine-tuned on [SciFact](https://github.com/allenai/scifact) and [Citation-Integrity](https://github.com/ScienceNLP-Lab/Citation-Integrity/)
            """
            )

    # Event listeners

    # Click button or press Enter to submit
    gr.on(
        triggers=[claim.submit, evidence.submit, submit.click],
        fn=predict,
        inputs=[claim, evidence],
        outputs=[prediction, label],
    )

    # Clear the previous predictions when a different model is selected
    gr.on(
        triggers=[dropdown.select],
        fn=lambda: "Model changed! Waiting for updated predictions...",
        outputs=[prediction],
    )

    # Update the predictions after changing the model
    dropdown.change(
        fn=select_model,
        inputs=dropdown,
    ).then(
        fn=predict,
        inputs=[claim, evidence],
        outputs=[prediction, label],
    )

    # Change visualization
    radio.change(fn=change_visualization, inputs=radio, outputs=[barplot, label])

demo.launch()
