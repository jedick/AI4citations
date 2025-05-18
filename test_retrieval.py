import unittest
import re
from retrieval import *

pdf_file = "work/Wikipedia.pdf"


class TestRetrieval(unittest.TestCase):

    def test_retrieve(self):
        evidence = retrieve_from_pdf(pdf_file, "Wikipedia is an encyclopedia", k=1)
        self.assertEqual(re.findall("encyclopedia", evidence), ["encyclopedia"])


if __name__ == "__main__":
    unittest.main()
