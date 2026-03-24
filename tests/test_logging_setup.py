"""Tests for centralized structured logging setup."""

import json
import logging

from src.config.logging import reset_logging, setup_logging


class TestSetupLogging:
    """Verify setup_logging configures the root logger correctly."""

    def setup_method(self):
        reset_logging()

    def teardown_method(self):
        reset_logging()

    def test_text_format_output(self, capfd):
        """Text format produces human-readable output."""
        setup_logging(level="INFO", fmt="text")
        logger = logging.getLogger("test.text")
        logger.info("hello world")
        captured = capfd.readouterr()
        assert "hello world" in captured.err
        assert "INFO" in captured.err

    def test_json_format_output(self, capfd):
        """JSON format produces valid JSON lines."""
        setup_logging(level="INFO", fmt="json")
        logger = logging.getLogger("test.json")
        logger.info("structured message")
        captured = capfd.readouterr()
        line = captured.err.strip()
        data = json.loads(line)
        assert data["message"] == "structured message"
        assert data["levelname"] == "INFO"

    def test_log_level_respected(self, capfd):
        """Messages below configured level are not emitted."""
        setup_logging(level="WARNING", fmt="text")
        logger = logging.getLogger("test.level")
        logger.info("should not appear")
        logger.warning("should appear")
        captured = capfd.readouterr()
        assert "should not appear" not in captured.err
        assert "should appear" in captured.err

    def test_idempotent_setup(self, capfd):
        """Calling setup_logging twice does not duplicate handlers."""
        setup_logging(level="INFO", fmt="text")
        setup_logging(level="INFO", fmt="text")  # second call is no-op
        root = logging.getLogger()
        assert len(root.handlers) == 1

    def test_json_includes_logger_name(self, capfd):
        """JSON output includes the logger name."""
        setup_logging(level="INFO", fmt="json")
        logger = logging.getLogger("mymodule.submodule")
        logger.info("test")
        captured = capfd.readouterr()
        data = json.loads(captured.err.strip())
        assert data["name"] == "mymodule.submodule"

    def test_otel_fields_empty_without_tracing(self, capfd):
        """Without active OTel span, trace_id/span_id are empty strings."""
        setup_logging(level="INFO", fmt="json")
        logger = logging.getLogger("test.otel")
        logger.info("no trace")
        captured = capfd.readouterr()
        data = json.loads(captured.err.strip())
        # OTel filter adds trace_id/span_id even when empty
        assert data.get("trace_id", "") == ""
