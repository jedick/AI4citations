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
    return {
        d["label"]: d["score"]
        for d in pipe({"text": evidence, "text_pair": claim}, top_k=3)
    }


# Function to select the model
def select_model(model_name):
    global pipe, MODEL_NAME
    MODEL_NAME = model_name
    pipe = pipeline(
        "text-classification",
        model=MODEL_NAME,
    )


def prediction_to_df(prediction=None):
    if prediction is None:
        # This shows a half-filled plot for app auto-reload
        prediction = {"SUPPORT": 0.5, "NEI": 0.5, "REFUTE": 0.5}
    elif prediction == "":
        # This shows an empty plot for app initialization
        prediction = {"SUPPORT": 0, "NEI": 0, "REFUTE": 0}
    elif "Model" in prediction:
        # This shows full-height bars when the model is changed
        prediction = {"SUPPORT": 1, "NEI": 1, "REFUTE": 1}
    else:
        # Convert predictions (text result from API) to dictionary
        prediction = eval(prediction)
        # Rename dictionary keys (different models have different labels)
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
    # This moves the index to the Class column
    return df.reset_index(names="Class")


# Gradio interface setup
with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column(scale=2, min_width=300):
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
            input_claim = gr.Textbox(label="Enter the claim (or hypothesis)")
            input_evidence = gr.Textbox(label="Enter the evidence (or premise)")
            prediction = gr.Textbox(label="Prediction")
            query_button = gr.Button("Submit")

        with gr.Column(scale=1, min_width=300):
            barplot = gr.BarPlot(
                prediction_to_df,
                x="Class",
                y="Probability",
                color="Class",
                color_map={"SUPPORT": "green", "NEI": "#888888", "REFUTE": "#FF8888"},
                inputs=prediction,
                y_lim=([0, 1]),
            )

    # Click button or press Enter to submit
    gr.on(
        triggers=[input_claim.submit, input_evidence.submit, query_button.click],
        fn=predict,
        inputs=[input_claim, input_evidence],
        outputs=[prediction],
    )

    # Clear the previous predictions as soon as a new model is selected
    # See https://www.gradio.app/guides/blocks-and-event-listeners
    def clear_prediction():
        return "Model changed! Waiting for updated predictions..."

    gr.on(
        triggers=[dropdown.select],
        fn=clear_prediction,
        outputs=[prediction],
    )

    # Update the predictions after changing the model
    dropdown.change(
        fn=select_model,
        inputs=dropdown,
    ).then(
        fn=predict,
        inputs=[input_claim, input_evidence],
        outputs=[prediction],
    )

demo.launch()
