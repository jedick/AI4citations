from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from litserve import LitAPI, LitServer


class PyversLitAPI(LitAPI):
    def setup(self, device, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        """
        Load the tokenizer and model, and move the model to the specified device.
        """
        self.device = device
        self.model_name = model_name

        # Instantiate tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # We have three classes for NLI (entailment, neutral, contradiction)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        )

        # Move model to the device (e.g., CPU, GPU)
        self.model.to(device)
        # Set the model in evaluation mode
        self.model.eval()

    def decode_request(self, request):
        """
        Process different requests:
            - Forward request to change the model
            - Preprocess the request data (tokenize)
        """
        if "model_name" in request:
            return request

        # Assuming request is a dictionary with a "evidence" and "claim" fields
        inputs = self.tokenizer(
            request["evidence"],
            request["claim"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return inputs

    def predict(self, inputs):
        """
        Act on the decoded inputs
        """
        if "model_name" in inputs:
            # Change the model if the given model name is not empty
            if not inputs["model_name"] == "":
                self.setup(self.device, inputs["model_name"])
        else:
            # Perform the inference
            with torch.no_grad():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
            return outputs

    def encode_response(self, outputs):
        """
        Process the model output into a response dictionary;
        Include the model name with every response
        """
        response = {}
        # If the model ran, its outputs are in a dictionary
        if isinstance(outputs, dict):
            # Convert logits to probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Return probability for each of three classes
            response = {
                "SUPPORT": probabilities[:, 0].item(),
                "NEI": probabilities[:, 1].item(),
                "REFUTE": probabilities[:, 2].item(),
            }

        return response, self.model_name
        # return response


if __name__ == "__main__":

    api = PyversLitAPI()
    server = LitServer(api, devices=1)
    server.run(port=8000)
