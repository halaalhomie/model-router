import unittest
from unittest import mock

from app.config import ConfigurationError, load_settings


COMPLETE_ENV = {
    "GEMINI_API_KEY": "test-key",
    "GEMINI_ANALYZER_MODEL": "analyzer",
    "GEMINI_FAST_MODEL": "fast",
    "GEMINI_CODE_MODEL": "code",
    "GEMINI_REASONING_MODEL": "reasoning",
    "GEMINI_LONG_CONTEXT_MODEL": "long-context",
}


class LoadSettingsTests(unittest.TestCase):
    """The developer's own .env is patched out so tests stay deterministic."""

    def setUp(self) -> None:
        patcher = mock.patch("app.config.load_dotenv")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_builds_settings_and_catalog_from_the_environment(self) -> None:
        with mock.patch.dict("os.environ", COMPLETE_ENV, clear=True):
            settings = load_settings()

        self.assertEqual(settings.analyzer_model, "analyzer")
        self.assertEqual(settings.catalog.fast_model, "fast")
        self.assertEqual(settings.catalog.code_model, "code")
        self.assertEqual(settings.catalog.reasoning_model, "reasoning")
        self.assertEqual(settings.catalog.long_context_model, "long-context")

    def test_names_the_variable_that_is_missing(self) -> None:
        for missing in COMPLETE_ENV:
            partial = {k: v for k, v in COMPLETE_ENV.items() if k != missing}

            with self.subTest(missing=missing):
                with mock.patch.dict("os.environ", partial, clear=True):
                    with self.assertRaises(ConfigurationError) as caught:
                        load_settings()

                self.assertIn(missing, str(caught.exception))
