"""MQTT publisher for the baby_in_crib state — feeds Home Assistant.

Runs as a background thread inside the hub process. Reads MQTT
settings from ``settings_store``; on enabled-and-broker-set, opens a
persistent MQTT connection (TCP or WebSocket per config) and
publishes:

  1. A Home Assistant MQTT Discovery payload, ONCE on each connect.
     Topic: ``<ha_discovery_prefix>/binary_sensor/<slug>/config``
     Body: standard HA binary_sensor discovery dict (occupancy
     device_class, state_topic, availability_topic, device metadata).
     This auto-creates ``binary_sensor.<slug>`` in HA.

  2. The current ON/OFF state to ``<topic_prefix>/baby_in_crib/state``
     whenever the operator-facing notion of "in crib" changes. Sent
     with retain=True so HA always has the latest value after restart.

  3. An availability "online"/"offline" string to
     ``<topic_prefix>/availability`` on connect/disconnect (LWT). HA
     greys out the sensor when the hub is down.

The hub calls :func:`set_baby_in_crib(state)` whenever the aggregate
flips; the publisher debounces (skip duplicate state).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Optional

try:
    import paho.mqtt.client as mqtt
    HAVE_PAHO = True
except ImportError:
    HAVE_PAHO = False


_log = logging.getLogger("mqtt_publisher")


def _slug(s: str) -> str:
    """Lowercase, alnum + underscore. Used for HA entity unique_id."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "_", s).strip("_")
    return s or "primeanalyze"


class MQTTPublisher:
    def __init__(self):
        self._settings: dict = {}
        self._client: Optional["mqtt.Client"] = None
        self._lock = threading.Lock()
        self._connected = False
        self._connecting = False
        self._last_state: Optional[bool] = None
        self._last_status_msg: str = "not configured"
        self._last_connect_t: float = 0.0
        self._stop = threading.Event()
        self._reconnect_thread: Optional[threading.Thread] = None

    # --- public API -----------------------------------------------------

    def reload(self, settings: dict) -> None:
        """Apply new settings. Reconnects if MQTT block changed."""
        with self._lock:
            self._settings = settings or {}
            mqtt_cfg = self._mqtt_cfg()
            enabled = bool(mqtt_cfg.get("enabled"))
            broker = (mqtt_cfg.get("broker") or "").strip()
            if not HAVE_PAHO:
                self._last_status_msg = "paho-mqtt not installed"
                self._teardown_client()
                return
            if not enabled:
                self._last_status_msg = "disabled"
                self._teardown_client()
                return
            if not broker:
                self._last_status_msg = "no broker set"
                self._teardown_client()
                return
            # Reconfigure: tear down + reconnect with new settings.
            self._teardown_client()
            self._spawn_reconnect_thread()

    def status(self) -> dict:
        """Snapshot for the /api/settings/mqtt/status endpoint."""
        with self._lock:
            return {
                "have_paho": HAVE_PAHO,
                "enabled": bool(self._mqtt_cfg().get("enabled")),
                "connected": self._connected,
                "last_status": self._last_status_msg,
                "last_connect_ts": self._last_connect_t,
                "last_state": self._last_state,
            }

    def set_baby_in_crib(self, state: bool) -> None:
        """Called from the state recorder thread. Debounced — only
        publishes on actual transitions."""
        state = bool(state)
        with self._lock:
            if state == self._last_state and self._connected:
                return
            self._last_state = state
            if not self._connected or self._client is None:
                return
            self._publish_state(state)

    def test_connect(self, settings_override: dict, timeout: float = 6.0) -> dict:
        """Synchronous one-shot connection test against the supplied
        settings (NOT applied to the running client). Returns a dict
        like ``{"ok": bool, "message": str}``."""
        if not HAVE_PAHO:
            return {"ok": False, "message": "paho-mqtt is not installed on the hub"}
        mqtt_cfg = (settings_override or {}).get("mqtt") or {}
        broker = (mqtt_cfg.get("broker") or "").strip()
        if not broker:
            return {"ok": False, "message": "broker host is empty"}
        port = int(mqtt_cfg.get("port") or 1883)
        transport = (mqtt_cfg.get("transport") or "tcp").strip()
        if transport not in ("tcp", "websockets"):
            return {"ok": False, "message": f"unknown transport {transport!r}"}
        try:
            cli = mqtt.Client(
                client_id=(mqtt_cfg.get("client_id") or "primeanalyze-test"),
                transport=transport,
                clean_session=True,
            )
        except TypeError:
            # paho-mqtt 2.x changed the constructor
            cli = mqtt.Client(  # type: ignore[call-arg]
                mqtt.CallbackAPIVersion.VERSION1,
                client_id=(mqtt_cfg.get("client_id") or "primeanalyze-test"),
                transport=transport,
                clean_session=True,
            )
        if transport == "websockets":
            cli.ws_set_options(path=(mqtt_cfg.get("ws_path") or "/mqtt"))
        if mqtt_cfg.get("use_tls"):
            cli.tls_set()
        username = mqtt_cfg.get("username") or ""
        password = mqtt_cfg.get("password") or ""
        if username:
            cli.username_pw_set(username, password or None)

        result: dict = {"ok": False, "message": "timeout"}
        done = threading.Event()

        def _on_connect(client, userdata, flags, rc, properties=None):
            if rc == 0:
                result["ok"] = True
                result["message"] = "connected"
            else:
                result["message"] = mqtt.connack_string(rc) if hasattr(mqtt, "connack_string") else f"rc={rc}"
            done.set()

        cli.on_connect = _on_connect
        try:
            cli.connect_async(broker, port, keepalive=15)
            cli.loop_start()
            done.wait(timeout=timeout)
        except Exception as e:
            result["message"] = f"{type(e).__name__}: {e}"
        finally:
            try:
                cli.loop_stop()
                cli.disconnect()
            except Exception:
                pass
        return result

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._teardown_client()

    # --- internals -------------------------------------------------------

    def _mqtt_cfg(self) -> dict:
        cfg = self._settings.get("mqtt") if isinstance(self._settings, dict) else None
        return cfg if isinstance(cfg, dict) else {}

    def _topic_state(self) -> str:
        prefix = (self._mqtt_cfg().get("topic_prefix") or "primeanalyze").rstrip("/")
        return f"{prefix}/baby_in_crib/state"

    def _topic_availability(self) -> str:
        prefix = (self._mqtt_cfg().get("topic_prefix") or "primeanalyze").rstrip("/")
        return f"{prefix}/availability"

    def _topic_discovery(self) -> str:
        cfg = self._mqtt_cfg()
        ha_prefix = (cfg.get("ha_discovery_prefix") or "homeassistant").rstrip("/")
        unique_id = _slug(cfg.get("device_name") or "primeanalyze_baby_in_crib")
        if not unique_id.startswith(_slug("baby_in_crib")):
            unique_id = f"{unique_id}_baby_in_crib"
        return f"{ha_prefix}/binary_sensor/{unique_id}/config"

    def _discovery_payload(self) -> dict:
        cfg = self._mqtt_cfg()
        device_name = cfg.get("device_name") or "PrimeAnalyze Baby Monitor"
        unique_id = _slug(device_name) + "_baby_in_crib"
        return {
            "name": "Baby in Crib",
            "unique_id": unique_id,
            "object_id": "baby_in_crib",  # → binary_sensor.baby_in_crib
            "state_topic": self._topic_state(),
            "availability_topic": self._topic_availability(),
            "payload_on": "ON",
            "payload_off": "OFF",
            "payload_available": "online",
            "payload_not_available": "offline",
            "device_class": "occupancy",
            "device": {
                "identifiers": [_slug(device_name)],
                "name": device_name,
                "model": "CV_Monitoring",
                "manufacturer": "PrimeAnalyze",
            },
        }

    def _spawn_reconnect_thread(self) -> None:
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return
        t = threading.Thread(target=self._reconnect_loop, name="mqtt-reconnect", daemon=True)
        self._reconnect_thread = t
        t.start()

    def _reconnect_loop(self) -> None:
        # Build the client and connect; on failure, retry with backoff.
        backoff = 2.0
        while not self._stop.is_set():
            with self._lock:
                cfg = self._mqtt_cfg()
                if not cfg.get("enabled"):
                    self._last_status_msg = "disabled"
                    return
                broker = (cfg.get("broker") or "").strip()
                if not broker:
                    self._last_status_msg = "no broker set"
                    return
            try:
                self._connect_now()
                return  # loop_start owns the connection from here
            except Exception as e:
                self._last_status_msg = f"connect failed: {e}"
                _log.warning("MQTT connect failed: %s — retry in %.0fs", e, backoff)
                if self._stop.wait(backoff):
                    return
                backoff = min(60.0, backoff * 1.7)

    def _connect_now(self) -> None:
        cfg = self._mqtt_cfg()
        broker = (cfg.get("broker") or "").strip()
        port = int(cfg.get("port") or 1883)
        transport = (cfg.get("transport") or "tcp").strip()
        client_id = (cfg.get("client_id") or "primeanalyze-hub").strip()

        try:
            cli = mqtt.Client(client_id=client_id, transport=transport, clean_session=False)
        except TypeError:
            cli = mqtt.Client(  # type: ignore[call-arg]
                mqtt.CallbackAPIVersion.VERSION1,
                client_id=client_id, transport=transport, clean_session=False,
            )
        if transport == "websockets":
            cli.ws_set_options(path=(cfg.get("ws_path") or "/mqtt"))
        if cfg.get("use_tls"):
            cli.tls_set()
        username = cfg.get("username") or ""
        password = cfg.get("password") or ""
        if username:
            cli.username_pw_set(username, password or None)
        # Last-Will so HA sees us go offline cleanly.
        cli.will_set(self._topic_availability(), payload="offline", qos=1, retain=True)
        cli.on_connect = self._on_connect
        cli.on_disconnect = self._on_disconnect
        cli.connect_async(broker, port, keepalive=30)
        cli.loop_start()
        with self._lock:
            self._client = cli
            self._connecting = True
            self._last_status_msg = f"connecting to {broker}:{port} ({transport})"

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc != 0:
            msg = mqtt.connack_string(rc) if hasattr(mqtt, "connack_string") else f"rc={rc}"
            with self._lock:
                self._connected = False
                self._connecting = False
                self._last_status_msg = f"reject: {msg}"
            _log.warning("MQTT reject: %s", msg)
            return
        with self._lock:
            self._connected = True
            self._connecting = False
            self._last_connect_t = time.time()
            self._last_status_msg = "connected"
            cfg = self._mqtt_cfg()
            # Availability online
            try:
                client.publish(self._topic_availability(), "online", qos=1, retain=True)
            except Exception:
                pass
            # Discovery
            if cfg.get("ha_discovery"):
                try:
                    client.publish(
                        self._topic_discovery(),
                        json.dumps(self._discovery_payload()),
                        qos=1,
                        retain=True,
                    )
                except Exception:
                    pass
            # Republish current state (if known)
            if self._last_state is not None:
                self._publish_state(self._last_state)
        _log.info("MQTT connected")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        with self._lock:
            self._connected = False
            self._last_status_msg = f"disconnected (rc={rc})"
        _log.info("MQTT disconnected (rc=%s)", rc)
        # paho's auto-reconnect kicks in via loop_start; nothing else needed.

    def _publish_state(self, state: bool) -> None:
        """Caller holds self._lock."""
        if self._client is None or not self._connected:
            return
        payload = "ON" if state else "OFF"
        try:
            self._client.publish(self._topic_state(), payload, qos=1, retain=True)
        except Exception as e:
            _log.warning("MQTT publish failed: %s", e)

    def _teardown_client(self) -> None:
        """Caller holds self._lock."""
        if self._client is not None:
            try:
                # Publish offline before going away.
                self._client.publish(self._topic_availability(), "offline", qos=1, retain=True)
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
        self._connected = False
        self._connecting = False


# Module-level singleton used by cv_hub + state_recorder.
publisher = MQTTPublisher()
