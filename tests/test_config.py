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
    "POSTGRES_USER": "router",
    "POSTGRES_PASSWORD": "secret",
    "POSTGRES_DB": "model_router",
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

    def test_defaults_the_request_timeout_when_unset(self) -> None:
        with mock.patch.dict("os.environ", COMPLETE_ENV, clear=True):
            settings = load_settings()

        self.assertEqual(settings.request_timeout_seconds, 30.0)

    def test_reads_the_request_timeout_override(self) -> None:
        env = dict(COMPLETE_ENV, REQUEST_TIMEOUT_SECONDS="45")

        with mock.patch.dict("os.environ", env, clear=True):
            settings = load_settings()

        self.assertEqual(settings.request_timeout_seconds, 45.0)

    def test_rejects_a_non_numeric_request_timeout(self) -> None:
        env = dict(COMPLETE_ENV, REQUEST_TIMEOUT_SECONDS="soon")

        with mock.patch.dict("os.environ", env, clear=True):
            with self.assertRaises(ConfigurationError):
                load_settings()

    def test_rejects_a_timeout_below_geminis_own_floor(self) -> None:
        """Gemini itself returns 400 'deadline too short' under 10s --
        caught here at startup instead of mid-request."""
        env = dict(COMPLETE_ENV, REQUEST_TIMEOUT_SECONDS="5")

        with mock.patch.dict("os.environ", env, clear=True):
            with self.assertRaises(ConfigurationError) as caught:
                load_settings()

        self.assertIn("REQUEST_TIMEOUT_SECONDS", str(caught.exception))

    def test_defaults_kafka_bootstrap_servers(self) -> None:
        with mock.patch.dict("os.environ", COMPLETE_ENV, clear=True):
            settings = load_settings()

        self.assertEqual(settings.kafka_bootstrap_servers, "localhost:9092")

    def test_reads_the_kafka_bootstrap_servers_override(self) -> None:
        env = dict(COMPLETE_ENV, KAFKA_BOOTSTRAP_SERVERS="broker-1:19092")

        with mock.patch.dict("os.environ", env, clear=True):
            settings = load_settings()

        self.assertEqual(settings.kafka_bootstrap_servers, "broker-1:19092")

    def test_derives_the_database_url_from_the_postgres_vars(self) -> None:
        """One source of truth: docker-compose creates the database from
        these same values, so the client URL is built from them rather
        than duplicating the password into a second variable."""
        with mock.patch.dict("os.environ", COMPLETE_ENV, clear=True):
            settings = load_settings()

        self.assertEqual(
            settings.database_url,
            "postgresql://router:secret@localhost:5432/model_router",
        )

    def test_database_host_and_port_can_be_overridden(self) -> None:
        env = dict(
            COMPLETE_ENV, POSTGRES_HOST="db.internal", POSTGRES_PORT="6543"
        )

        with mock.patch.dict("os.environ", env, clear=True):
            settings = load_settings()

        self.assertEqual(
            settings.database_url,
            "postgresql://router:secret@db.internal:6543/model_router",
        )

    def test_an_explicit_database_url_wins(self) -> None:
        """What a deployment against a managed database would set."""
        env = dict(COMPLETE_ENV, DATABASE_URL="postgresql://elsewhere/db")

        with mock.patch.dict("os.environ", env, clear=True):
            settings = load_settings()

        self.assertEqual(settings.database_url, "postgresql://elsewhere/db")

    def test_accepts_a_timeout_exactly_at_the_floor(self) -> None:
        env = dict(COMPLETE_ENV, REQUEST_TIMEOUT_SECONDS="10")

        with mock.patch.dict("os.environ", env, clear=True):
            settings = load_settings()  # must not raise

        self.assertEqual(settings.request_timeout_seconds, 10.0)

    def test_names_the_variable_that_is_missing(self) -> None:
        for missing in COMPLETE_ENV:
            partial = {k: v for k, v in COMPLETE_ENV.items() if k != missing}

            with self.subTest(missing=missing):
                with mock.patch.dict("os.environ", partial, clear=True):
                    with self.assertRaises(ConfigurationError) as caught:
                        load_settings()

                self.assertIn(missing, str(caught.exception))
