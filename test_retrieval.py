import unittest
import re
from retrieval import *

pdf_file = "examples/retrieval/CRISPR.pdf"


class TestRetrieval(unittest.TestCase):

    def test_retrieve(self):
        evidence = retrieve_from_pdf(
            pdf_file, "CRISPR evolution has been described as Lamarckian", k=1
        )
        self.assertEqual(re.findall("Koonin", evidence), ["Koonin"])


if __name__ == "__main__":
    unittest.main()
