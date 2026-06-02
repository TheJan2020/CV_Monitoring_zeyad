"""Read / write the hub's settings.json (MQTT credentials etc).

Single source of truth for non-camera, non-secret configuration that
the operator can edit at runtime via the Settings page. Schema is a
plain dict; this module only knows how to load and save it safely
(atomic rename + utf-8 without BOM, same convention as cameras.json).

Default schema:

  {
    "mqtt": {
      "enabled": false,
      "broker": "",
      "port": 1883,
      "transport": "tcp",            # "tcp" | "websockets"
      "ws_path": "/mqtt",            # only used when transport=websockets
      "use_tls": false,
      "username": "",
      "password": "",
      "client_id": "primeanalyze-hub",
      "topic_prefix": "primeanalyze",
      "ha_discovery": true,
      "ha_discovery_prefix": "homeassistant",
      "device_name": "PrimeAnalyze Baby Monitor"
    }
  }
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


_PATH = Path(__file__).resolve().parent / "config" / "settings.json"

_DEFAULTS: dict[str, Any] = {
    "mqtt": {
        "enabled": False,
        "broker": "",
        "port": 1883,
        "transport": "tcp",
        "ws_path": "/mqtt",
        "use_tls": False,
        "username": "",
        "password": "",
        "client_id": "primeanalyze-hub",
        "topic_prefix": "primeanalyze",
        "ha_discovery": True,
        "ha_discovery_prefix": "homeassistant",
        "device_name": "PrimeAnalyze Baby Monitor",
    },
}


def _merge_defaults(loaded: dict, defaults: dict) -> dict:
    """Deep-merge defaults under loaded (loaded wins on conflict)."""
    out: dict = {}
    for k, v in defaults.items():
        if isinstance(v, dict) and isinstance(loaded.get(k), dict):
            out[k] = _merge_defaults(loaded.get(k, {}), v)
        elif k in loaded:
            out[k] = loaded[k]
        else:
            out[k] = v
    # Carry over any extra keys the loaded file had (forward-compat).
    for k, v in loaded.items():
        if k not in out:
            out[k] = v
    return out


def load() -> dict:
    """Load settings, filling defaults for any missing keys. Never
    raises — returns the defaults on any I/O / parse error."""
    try:
        text = _PATH.read_text(encoding="utf-8-sig")
        loaded = json.loads(text) if text.strip() else {}
    except FileNotFoundError:
        return json.loads(json.dumps(_DEFAULTS))
    except (OSError, json.JSONDecodeError):
        return json.loads(json.dumps(_DEFAULTS))
    return _merge_defaults(loaded if isinstance(loaded, dict) else {}, _DEFAULTS)


def save(settings: dict) -> None:
    """Atomic write: temp file in the same directory + os.replace."""
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(settings, indent=2, sort_keys=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".settings.", suffix=".tmp", dir=str(_PATH.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, _PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update_section(section: str, patch: dict) -> dict:
    """Merge ``patch`` into ``settings[section]`` and persist. Returns
    the new full settings dict."""
    current = load()
    section_now = current.get(section)
    if not isinstance(section_now, dict):
        section_now = {}
    section_now.update(patch)
    current[section] = section_now
    save(current)
    return current
