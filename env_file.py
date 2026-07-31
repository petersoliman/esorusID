"""Load configuration from a .env file sitting beside this module.

Every setting here — which catalog to talk to, where its images live, the API
key — was read from the process environment only. That works when a human
exports them before running a command, and fails silently the moment the
service restarts under a supervisor that does not: the app comes back up
without a catalog, and search results quietly lose their supplier names with
nothing in the logs to say why. That is exactly what happened in production.

Reading a file that lives next to the code makes the configuration a property
of the deployment rather than of whoever last started the process.

Real environment variables always win, so a systemd unit, a container, or an
`export` on the command line can still override the file — the file is the
fallback, not the authority.
"""

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"

_loaded = False


def load_env(path: Path = None) -> int:
    """Merge KEY=VALUE lines from `path` into os.environ.

    Returns the number of variables set. Missing file is not an error — the
    process environment may already carry everything needed.

    Safe to call repeatedly; only the first call reads the file.
    """
    global _loaded
    if _loaded:
        return 0
    _loaded = True

    target = Path(path) if path else ENV_PATH
    if not target.is_file():
        return 0

    applied = 0
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip one layer of matching quotes, so a value with spaces or a
        # trailing comment marker can be written naturally.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        applied += 1

    return applied
