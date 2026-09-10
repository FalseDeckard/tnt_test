import re
from collections.abc import Iterable

WORD_PATTERN = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
RUSSIAN_STOP_WORDS = frozenset(
    """
    а без более больше будет будто бы был была были было быть в вам вас весь во
    вот все всего всех вы где да даже для до его ее если есть еще же за здесь и
    из или им их к как какой когда ко кто ли либо мне может мы на над надо наш не
    него нее нет ни них но ну о об однако он она они оно от очень по под при про
    раз с сам сама сами со так такой там те тем то того тоже той только том ты у
    уже хотя чего чей чем что чтобы чье чья эта эти это этот я
    """.split()
)


def tokenize_words(text: str) -> list[str]:
    """Return lowercase word tokens without requiring downloaded resources."""
    if not isinstance(text, str):
        return []
    return WORD_PATTERN.findall(text.lower())


class TextPreprocessor:
    """Shared Russian text cleanup and lemmatization for indexing and queries."""

    def __init__(self, morph=None, stop_words: Iterable[str] = RUSSIAN_STOP_WORDS):
        if morph is None:
            import pymorphy3

            morph = pymorphy3.MorphAnalyzer()
        self.morph = morph
        self.stop_words = frozenset(stop_words)

    @staticmethod
    def basic_clean(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return " ".join(text.split())

    def full_clean(self, text: str) -> str:
        tokens = tokenize_words(self.basic_clean(text))
        lemmas = (
            self.morph.parse(token)[0].normal_form
            for token in tokens
            if token not in self.stop_words
        )
        return " ".join(lemmas)
