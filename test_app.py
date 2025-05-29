import unittest
import pandas
from app import *


class TestApp(unittest.TestCase):

    def test_prediction_to_df(self):
        prediction = '{"Support": 0, "NEI": 0, "Refute": 0}'
        self.assertEqual(
            type(prediction_to_df(prediction)), pandas.core.frame.DataFrame
        )


if __name__ == "__main__":
    unittest.main()
