import unittest
import subprocess
import time
from gradio_client import Client


class TestApp(unittest.TestCase):
    """
    Integration tests for AI4citations app
    """

    def setUp(self):

        # Start the Gradio app in a separate process
        self.process = subprocess.Popen(["python", "app.py"])
        # Wait for the server to start
        time.sleep(60)

    def tearDown(self):
        # Terminate the Gradio process
        self.process.terminate()
        self.process.wait()

    def test_app(self):
        # Setup the Gradio client
        client = Client("http://127.0.0.1:7860/")
        # Make a few easy predictions
        result_support = client.predict(
            claim="Yes", evidence="Yes", api_name="/query_model"
        )
        result_nei = client.predict(
            claim="Yes", evidence="Maybe", api_name="/query_model"
        )
        result_refute = client.predict(
            claim="Yes", evidence="No", api_name="/query_model"
        )
        self.assertEqual(result_support["label"], "SUPPORT")
        self.assertEqual(result_nei["label"], "NEI")
        self.assertEqual(result_refute["label"], "REFUTE")


if __name__ == "__main__":
    unittest.main()
