import unittest
import re
from retrieval_bm25s import *

pdf_file = "examples/retrieval/CRISPR.pdf"


class TestRetrieval(unittest.TestCase):

    def test_retrieve(self):
        evidence = retrieve_with_bm25s(
            pdf_file, "CRISPR evolution has been described as Lamarckian", top_k=1
        )
        self.assertEqual(re.findall("Koonin", evidence), ["Koonin"])


if __name__ == "__main__":
    unittest.main()
