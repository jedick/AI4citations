import gradio as gr
import requests
import pandas as pd


# Function to query the model
def query_model(claim, evidence):
    url = "http://127.0.0.1:8000/predict"
    data = {
        "claim": claim,
        "evidence": evidence,
    }
    response = requests.post(url, json=data)
    # Return the predictions
    return response.json()[0]


# Function to select the model
def select_model(model_name):
    url = "http://127.0.0.1:8000/predict"
    data = {"model_name": model_name}
    response = requests.post(url, json=data)


# Function to get the current model
def get_model():
    url = "http://127.0.0.1:8000/predict"
    data = {"model_name": ""}
    response = requests.post(url, json=data)
    # Return the model name
    return response.json()[1]


# Use global state so correct model is shown after page refresh
# https://github.com/gradio-app/gradio/issues/3173
state = get_model()


def update_state(model_name):
    global state
    state = model_name


# Gradio interface setup
with gr.Blocks() as app:

    with gr.Row():
        with gr.Column(scale=2, min_width=300):
            # Create dropdown menu to select the model
            dropdown = gr.Dropdown(
                label="Select Model",
                # On initialization, get the current model name from the server
                value=lambda: state,
                choices=[
                    "bert-base-uncased",
                    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                    "jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint",
                ],
            )
            input_claim = gr.Textbox(label="Enter the claim (or hypothesis)")
            input_evidence = gr.Textbox(label="Enter the evidence (or premise)")
            classification = gr.Textbox(label="Classification")
            query_button = gr.Button("Submit")

        def classification_to_df(classification=None):
            if classification == "":
                # This shows an empty plot for app initialization
                class_dict = {"SUPPORT": 0, "NEI": 0, "REFUTE": 0}
            elif "Model" in classification:
                # This shows full-height bars when the model is changed
                class_dict = {"SUPPORT": 1, "NEI": 1, "REFUTE": 1}
            else:
                # Convert classifications (text result from API) to dictionary
                class_dict = eval(classification)
            # Convert dictionary to DataFrame with one column (Probability)
            df = pd.DataFrame.from_dict(
                class_dict, orient="index", columns=["Probability"]
            )
            # This moves the index to the Class column
            return df.reset_index(names="Class")

        with gr.Column(scale=1, min_width=300):
            barplot = gr.BarPlot(
                classification_to_df,
                x="Class",
                y="Probability",
                color="Class",
                color_map={"SUPPORT": "green", "NEI": "#888888", "REFUTE": "#FF8888"},
                inputs=classification,
                y_lim=([0, 1]),
            )

    # Click button or press Enter to submit
    gr.on(
        triggers=[input_claim.submit, input_evidence.submit, query_button.click],
        fn=query_model,
        inputs=[input_claim, input_evidence],
        outputs=[classification],
    )

    # Clear the previous predictions as soon as a new model is selected
    # See https://www.gradio.app/guides/blocks-and-event-listeners
    def clear_classification():
        return "Model changed! Waiting for updated predictions..."

    gr.on(
        triggers=[dropdown.select],
        fn=clear_classification,
        outputs=[classification],
    )

    # Update the predictions after changing the model
    dropdown.change(
        fn=select_model,
        inputs=dropdown,
    ).then(
        fn=query_model,
        inputs=[input_claim, input_evidence],
        outputs=[classification],
    )

    # Also update the global state with the model name
    dropdown.change(update_state, dropdown, None)

app.launch()
