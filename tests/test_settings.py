"""Tests for centralized Pydantic Settings configuration."""

import pytest
from pathlib import Path

from src.config.settings import (
    Settings,
    get_settings,
    _load_yaml,
    _CONFIG_PATH,
)


class TestSettingsDefaults:
    """Verify field defaults when no YAML or env vars are present."""

    def test_default_search_strategy(self) -> None:
        s = Settings()
        assert s.retrieval.search_strategy == "hybrid"

    def test_default_hybrid_alpha(self) -> None:
        s = Settings()
        assert s.retrieval.hybrid.alpha == 0.5

    def test_default_weaviate_host(self) -> None:
        s = Settings()
        assert s.weaviate.host == "localhost"
        assert s.weaviate.port == 8080

    def test_default_logging(self) -> None:
        s = Settings()
        assert s.logging.level == "INFO"
        assert s.logging.format == "text"

    def test_default_llm(self) -> None:
        s = Settings()
        assert s.llm.num_ctx == 4096
        assert s.llm.num_predict == 1024


class TestYAMLLoading:
    """Verify YAML config loading."""

    def test_load_yaml_with_real_config(self) -> None:
        data = _load_yaml(_CONFIG_PATH)
        assert data.get("retrieval", {}).get("search_strategy") == "hybrid"

    def test_load_yaml_missing_file(self, tmp_path: Path) -> None:
        data = _load_yaml(tmp_path / "nonexistent.yaml")
        assert data == {}

    def test_from_yaml_loads_config(self) -> None:
        s = Settings.from_yaml()
        assert s.retrieval.search_strategy == "hybrid"
        assert s.reranking.enabled is True
        assert s.llm.model == "llama3.2:3b"


class TestEnvironmentOverrides:
    """Verify env vars override YAML and defaults."""

    def test_nested_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RETRIEVAL__SEARCH_STRATEGY", "keyword")
        s = Settings()
        assert s.retrieval.search_strategy == "keyword"

    def test_nested_env_override_deep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RETRIEVAL__HYBRID__ALPHA", "0.8")
        s = Settings()
        assert s.retrieval.hybrid.alpha == 0.8


class TestValidation:
    """Verify Pydantic validation rejects invalid values."""

    def test_invalid_search_strategy(self) -> None:
        with pytest.raises(Exception):
            Settings(retrieval={"search_strategy": "invalid"})

    def test_invalid_alpha_range(self) -> None:
        with pytest.raises(Exception):
            Settings(retrieval={"hybrid": {"alpha": 1.5}})

    def test_invalid_log_format(self) -> None:
        with pytest.raises(Exception):
            Settings(logging={"format": "xml"})


class TestGetSettingsCache:
    """Verify singleton caching behavior."""

    def test_get_settings_returns_same_instance(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_reloads(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        assert s1 is not s2
        assert s1.retrieval.search_strategy == s2.retrieval.search_strategy
