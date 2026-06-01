"""XDG-compliant default paths for TaxoTreeSet.

Resolves where the tool stores its data, state, and configuration when
the user does not override the locations on the command line. The
resolution follows the XDG Base Directory Specification so that an
installation via ``pip install`` behaves like a well-mannered
user-level tool: it writes under the user's home (or the XDG_* override
locations) and never requires root or system directories such as
``/var``.

Resolved locations (with their XDG_* overrides):

- Vault (genome sequences, the registry) -> ``$XDG_DATA_HOME/taxotreeset``
  (default ``~/.local/share/taxotreeset``).
- Logs and run state -> ``$XDG_STATE_HOME/taxotreeset``
  (default ``~/.local/state/taxotreeset``).
- Configuration -> ``$XDG_CONFIG_HOME/taxotreeset``
  (default ``~/.config/taxotreeset``).

Generated datasets are the user-facing product and are not placed here;
they default to the current working directory and are set via the
``--output`` option of the relevant command.
"""
import os
from pathlib import Path

_APP_NAME = "taxotreeset"


def _xdg_base(env_var: str, default_subpath: str) -> Path:
    """Resolve an XDG base directory, honoring the environment override.

    Args:
        env_var: The XDG environment variable to consult (e.g.
            ``"XDG_DATA_HOME"``).
        default_subpath: Path under the user's home to fall back to when
            the variable is unset or empty (e.g. ``".local/share"``).

    Returns:
        The resolved base directory as a Path. The application
        subdirectory is not appended here.
    """
    value = os.environ.get(env_var, "").strip()
    if value:
        return Path(value)
    return Path.home() / default_subpath


def data_dir(create: bool = True) -> Path:
    """Return the per-user data directory for the vault and registry.

    Args:
        create: When True, create the directory (and parents) if absent.

    Returns:
        ``$XDG_DATA_HOME/taxotreeset`` or the default under ~/.local/share.
    """
    path = _xdg_base("XDG_DATA_HOME", ".local/share") / _APP_NAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def state_dir(create: bool = True) -> Path:
    """Return the per-user state directory for logs and run state.

    Args:
        create: When True, create the directory (and parents) if absent.

    Returns:
        ``$XDG_STATE_HOME/taxotreeset`` or the default under ~/.local/state.
    """
    path = _xdg_base("XDG_STATE_HOME", ".local/state") / _APP_NAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def config_dir(create: bool = True) -> Path:
    """Return the per-user configuration directory.

    Args:
        create: When True, create the directory (and parents) if absent.

    Returns:
        ``$XDG_CONFIG_HOME/taxotreeset`` or the default under ~/.config.
    """
    path = _xdg_base("XDG_CONFIG_HOME", ".config") / _APP_NAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def default_vault_path() -> Path:
    """Return the default LMDB vault directory under the data dir."""
    return data_dir() / "vault"


def default_registry_path() -> Path:
    """Return the default registry JSON path under the data dir."""
    return data_dir() / "registry.json"


def log_path(filename: str) -> Path:
    """Return a path for a log file under the state directory.

    Args:
        filename: Log file name, e.g. ``"generation.log"``.

    Returns:
        Full path to the log file under ``$XDG_STATE_HOME/taxotreeset``.
    """
    return state_dir() / filename
