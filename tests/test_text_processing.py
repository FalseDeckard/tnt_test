import unittest

from app.core.text_processing import TextPreprocessor, tokenize_words


class ParsedWord:
    def __init__(self, word):
        self.normal_form = word.rstrip("ы")


class FakeMorph:
    def parse(self, word):
        return [ParsedWord(word)]


class TextProcessingTests(unittest.TestCase):
    def setUp(self):
        self.processor = TextPreprocessor(morph=FakeMorph(), stop_words={"и", "в"})

    def test_tokenize_words_keeps_unicode_words_and_hyphens(self):
        self.assertEqual(
            tokenize_words("Ёлки-палки, news_24 и 2026!"),
            ["ёлки-палки", "news", "и"],
        )

    def test_full_clean_filters_stop_words_and_punctuation(self):
        self.assertEqual(
            self.processor.full_clean("Коты и собаки — в домах."),
            "кот собаки домах",
        )

    def test_non_string_input_is_empty(self):
        self.assertEqual(self.processor.basic_clean(None), "")
        self.assertEqual(self.processor.full_clean(None), "")


if __name__ == "__main__":
    unittest.main()
