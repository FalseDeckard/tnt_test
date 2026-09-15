import unittest

from app.core.model_config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    MODEL_NAME,
    MODEL_REVISION,
)


class ModelConfigTests(unittest.TestCase):
    def test_default_model_is_pinned_to_a_commit(self):
        self.assertEqual(DEFAULT_MODEL_NAME, "deepvk/USER-bge-m3")
        self.assertRegex(DEFAULT_MODEL_REVISION, r"^[0-9a-f]{40}$")

    def test_effective_model_configuration_is_not_empty(self):
        self.assertTrue(MODEL_NAME)
        self.assertTrue(MODEL_REVISION)


if __name__ == "__main__":
    unittest.main()
