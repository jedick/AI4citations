import unittest
import pandas
from app import *


class TestApp(unittest.TestCase):

    def test_prediction_to_df(self):
        prediction = '{"SUPPORT": 0, "NEI": 0, "REFUTE": 0}'
        self.assertEqual(
            type(prediction_to_df(prediction)), pandas.core.frame.DataFrame
        )


if __name__ == "__main__":
    unittest.main()
