import unittest
import os
from ASS_Search import normalize_url, SearchIndex

class TestASS(unittest.TestCase):

    def test_normalize_url(self):
        self.assertEqual(normalize_url("http://example.com"), "https://example.com/")
        self.assertEqual(normalize_url("https://example.com/path?a=1&b=2"), "https://example.com/path?a=1&b=2")
        self.assertEqual(normalize_url("https://example.com/path?b=2&a=1"), "https://example.com/path?a=1&b=2")
        self.assertEqual(normalize_url("https://example.com/path/index.html"), "https://example.com/path/")
        self.assertEqual(normalize_url("https://example.com/path#fragment"), "https://example.com/path")

    def test_search_index(self):
        db_file = "test.db"
        if os.path.exists(db_file):
            os.remove(db_file)

        index = SearchIndex(db_file)
        index.index_document("https://example.com", {"title": "Example Domain", "snippet": "This is an example."})
        
        results, total = index.search("example")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.com")
        self.assertEqual(results[0]["title"], "Example Domain")

        os.remove(db_file)

    def test_query_language(self):
        db_file = "test.db"
        if os.path.exists(db_file):
            os.remove(db_file)

        index = SearchIndex(db_file)
        index.index_document("https://example.com/1", {"title": "First", "snippet": "This is the first document."})
        index.index_document("https://example.com/2", {"title": "Second", "snippet": "This is the second document."})
        index.index_document("https://example.com/3", {"title": "Third", "snippet": "This is the third document, which is special."})

        results, total = index.search("document")
        self.assertEqual(len(results), 3)

        results, total = index.search("document -second")
        self.assertEqual(len(results), 2)

        results, total = index.search('"third document"')
        self.assertEqual(len(results), 1)

        os.remove(db_file)

if __name__ == '__main__':
    unittest.main()
