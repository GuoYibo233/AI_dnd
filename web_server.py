#!/usr/bin/env python3
"""轻量级 Flask Web 入口，用于在浏览器中驱动 DMEngine。"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, make_response, request

from dm_engine import DMEngine

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STORY_ID = os.environ.get("STORY_ID", "emberlight_demo_v1")
STORY_DIR = Path(os.environ.get("STORY_PATH") or (BASE_DIR / "stories" / DEFAULT_STORY_ID)).resolve()
if not STORY_DIR.exists():
    raise FileNotFoundError(f"故事目录不存在：{STORY_DIR}")

app = Flask(__name__, static_folder="web_client", static_url_path="")

_engine_lock = threading.Lock()
_engine: Optional[DMEngine] = None
_opening_text: str = ""


def _create_engine(dialogue_id: Optional[str] = None) -> DMEngine:
    return DMEngine(str(STORY_DIR), conversation_id=dialogue_id)


def _reset_engine_locked(dialogue_id: Optional[str] = None) -> dict:
    global _engine, _opening_text
    engine = _create_engine(dialogue_id)
    opening = engine.play_opening_scene() or ""
    _engine = engine
    _opening_text = opening
    return {
        "opening": opening,
        "session_id": engine.session_id,
        "turn_counter": engine.turn_counter,
        "player_name": engine.player_display_name,
    }


def _ensure_engine_locked() -> DMEngine:
    global _engine
    if _engine is None:
        _reset_engine_locked()
    assert _engine is not None
    return _engine


@app.route("/")
def index() -> str:
    return app.send_static_file("index.html")


def _corsify(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def _preflight_response():
    return _corsify(make_response(("", 204)))


@app.after_request
def _apply_cors(response):
    if request.path.startswith("/api/"):
        return _corsify(response)
    return response


@app.route("/api/session", methods=["GET", "OPTIONS"])
def session_state():
    if request.method == "OPTIONS":
        return _preflight_response()
    with _engine_lock:
        if _engine is None:
            payload = {
                "initialized": False,
                "opening": "",
                "turn_counter": 0,
                "current_location": None,
                "active_npcs": [],
                "session_id": None,
                "player_name": None,
            }
        else:
            engine = _engine
            payload = {
                "initialized": True,
                "opening": _opening_text,
                "turn_counter": engine.turn_counter,
                "current_location": engine.current_location,
                "active_npcs": list(engine.active_contexts.keys()),
                "session_id": engine.session_id,
                "player_name": engine.player_display_name,
            }
    return jsonify(payload)


@app.route("/api/session/start", methods=["POST", "OPTIONS"])
def start_session():
    if request.method == "OPTIONS":
        return _preflight_response()
    payload_json = request.get_json(silent=True) or {}
    dialogue_id = payload_json.get("dialogue_id")
    with _engine_lock:
        payload = _reset_engine_locked(dialogue_id)
    return jsonify({"message": "session started", **payload})


@app.route("/api/reset", methods=["POST", "OPTIONS"])
def reset_session():
    if request.method == "OPTIONS":
        return _preflight_response()
    payload_json = request.get_json(silent=True) or {}
    dialogue_id = payload_json.get("dialogue_id")
    with _engine_lock:
        payload = _reset_engine_locked(dialogue_id)
    return jsonify(payload)


@app.route("/api/session/history", methods=["GET", "OPTIONS"])
def session_history():
    if request.method == "OPTIONS":
        return _preflight_response()
    with _engine_lock:
        if _engine is None:
            return jsonify({"session_id": None, "entries": [], "turn_counter": 0, "player_name": None})
        engine = _engine
        entries = engine.get_transcript_entries()
        payload = {
            "session_id": engine.session_id,
            "entries": entries,
            "turn_counter": engine.turn_counter,
            "player_name": engine.player_display_name,
        }
    return jsonify(payload)


@app.route("/api/turn", methods=["POST", "OPTIONS"])
def process_turn():
    if request.method == "OPTIONS":
        return _preflight_response()
    data = request.get_json(silent=True) or {}
    player_text = (data.get("player_text") or "").strip()
    if not player_text:
        return jsonify({"error": "player_text 不能为空"}), 400

    with _engine_lock:
        if _engine is None:
            return jsonify({"error": "会话尚未开始，请先启动剧本"}), 400
        engine = _ensure_engine_locked()
        try:
            result = engine.process_turn(player_text)
            result["turn_counter"] = engine.turn_counter
            result["session_id"] = engine.session_id
            result["player_name"] = engine.player_display_name
        except Exception as exc:  # pragma: no cover - 调试信息
            app.logger.exception("DMEngine 处理失败")
            return jsonify({"error": str(exc)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
