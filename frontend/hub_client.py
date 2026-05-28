"""Thin HTTP client for the cv_hub.py service on localhost.

The hub trusts requests originating from 127.0.0.1, so no session dance
is needed when the frontend and hub run on the same machine.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

_HUB_URL = os.environ.get("HUB_URL", "http://127.0.0.1:8000").rstrip("/")


def _request(method: str, path: str, body=None, timeout: float = 10.0):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        _HUB_URL + path, data=data, method=method, headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read()
            return {"_status": r.status, "_body": json.loads(text) if text else None}
    except urllib.error.HTTPError as e:
        body_text = b""
        try:
            body_text = e.read()
        except Exception:
            pass
        try:
            body_json = json.loads(body_text) if body_text else {}
        except Exception:
            body_json = {"error": body_text.decode("utf-8", "replace")}
        return {"_status": e.code, "_body": body_json}
    except urllib.error.URLError as e:
        return {"_status": 0, "_body": {"error": f"hub unreachable: {e.reason}"}}


def list_cameras() -> list[dict]:
    res = _request("GET", "/api/cameras")
    body = res.get("_body")
    return body if isinstance(body, list) else []


def get_camera(cam_id: str) -> dict | None:
    for c in list_cameras():
        if c.get("id") == cam_id:
            return c
    return None


def create_camera(payload: dict) -> tuple[int, dict]:
    res = _request("POST", "/api/cameras", body=payload, timeout=30.0)
    return res["_status"], res["_body"] or {}


def update_camera(cam_id: str, payload: dict) -> tuple[int, dict]:
    res = _request("PUT", f"/api/cameras/{cam_id}", body=payload, timeout=30.0)
    return res["_status"], res["_body"] or {}


def delete_camera(cam_id: str) -> tuple[int, dict]:
    res = _request("DELETE", f"/api/cameras/{cam_id}", timeout=30.0)
    return res["_status"], res["_body"] or {}


def test_rtsp(rtsp_url: str) -> tuple[int, dict]:
    res = _request(
        "POST", "/api/cameras/test", body={"rtsp_url": rtsp_url}, timeout=15.0,
    )
    return res["_status"], res["_body"] or {}


def get_state(cam_id: str) -> tuple[int, dict]:
    res = _request("GET", f"/api/state/{cam_id}", timeout=3.0)
    return res["_status"], res["_body"] or {}


def get_health() -> tuple[int, dict]:
    res = _request("GET", "/healthz", timeout=3.0)
    return res["_status"], res["_body"] or {}
