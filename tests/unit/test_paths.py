"""Tests for taxotreeset.paths — XDG-compliant path resolution."""

from pathlib import Path

from taxotreeset.paths import (
    _xdg_base,
    config_dir,
    data_dir,
    default_registry_path,
    default_vault_path,
    log_path,
    state_dir,
)


# ---------------------------------------------------------------------------
# _xdg_base
# ---------------------------------------------------------------------------


class TestXdgBase:
    def test_returns_env_var_path_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        result = _xdg_base("XDG_DATA_HOME", ".local/share")
        assert result == tmp_path

    def test_returns_home_subpath_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        result = _xdg_base("XDG_DATA_HOME", ".local/share")
        assert result == Path.home() / ".local/share"

    def test_empty_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", "")
        result = _xdg_base("XDG_DATA_HOME", ".local/share")
        assert result == Path.home() / ".local/share"

    def test_whitespace_only_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", "   ")
        result = _xdg_base("XDG_DATA_HOME", ".local/share")
        assert result == Path.home() / ".local/share"


# ---------------------------------------------------------------------------
# data_dir
# ---------------------------------------------------------------------------


class TestDataDir:
    def test_default_path_ends_with_taxotreeset(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        result = data_dir(create=False)
        assert result.name == "taxotreeset"
        assert result.parent == Path.home() / ".local/share"

    def test_honors_xdg_data_home_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        result = data_dir(create=False)
        assert result == tmp_path / "taxotreeset"

    def test_create_true_creates_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        result = data_dir(create=True)
        assert result.exists()
        assert result.is_dir()

    def test_create_false_does_not_create_directory(self, monkeypatch, tmp_path):
        override = tmp_path / "nonexistent_xdg"
        monkeypatch.setenv("XDG_DATA_HOME", str(override))
        result = data_dir(create=False)
        assert not result.exists()


# ---------------------------------------------------------------------------
# state_dir
# ---------------------------------------------------------------------------


class TestStateDir:
    def test_default_path_is_under_local_state(self, monkeypatch):
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        result = state_dir(create=False)
        assert result == Path.home() / ".local/state" / "taxotreeset"

    def test_honors_xdg_state_home_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        result = state_dir(create=False)
        assert result == tmp_path / "taxotreeset"


# ---------------------------------------------------------------------------
# config_dir
# ---------------------------------------------------------------------------


class TestConfigDir:
    def test_default_path_is_under_config(self, monkeypatch):
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        result = config_dir(create=False)
        assert result == Path.home() / ".config" / "taxotreeset"

    def test_honors_xdg_config_home_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        result = config_dir(create=False)
        assert result == tmp_path / "taxotreeset"

    def test_create_true_creates_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        result = config_dir(create=True)
        assert result.exists()
        assert result.is_dir()


# ---------------------------------------------------------------------------
# derived paths
# ---------------------------------------------------------------------------


class TestDerivedPaths:
    def test_default_vault_path_is_under_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        result = default_vault_path()
        assert result == tmp_path / "taxotreeset" / "vault"

    def test_default_registry_path_is_under_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        result = default_registry_path()
        assert result == tmp_path / "taxotreeset" / "registry.json"

    def test_log_path_is_under_state_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        result = log_path("generation.log")
        assert result == tmp_path / "taxotreeset" / "generation.log"

    def test_log_path_accepts_arbitrary_filename(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        result = log_path("discovery_2024.log")
        assert result.name == "discovery_2024.log"
