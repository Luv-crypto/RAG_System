

import os, uuid, datetime, json, markdown2
from pathlib import Path
from typing import List, Tuple

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, make_response
)

# ---------- your RAG core (imported) ---------------------------
from rag_scipdf_core import smart_query   # <- must be importable!

# ---------- constants ------------------------------------------
ROOT               = Path(__file__).parent.resolve()
OBJ_DIR_IMG        = ROOT / "object_store" / "images"
OBJ_DIR_TBL        = ROOT / "object_store" / "tables"
SESSION_COOKIE_KEY = "sid"

# ---------- Flask -------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "hd39u4h3j4iejfioj3948jf0394jf0394jf0j394jf0394"

# ---------- in-memory session store ------------------------------
# { session_id : [ {role, html, ts}, ... ] }
CHAT_LOGS = {}


# ---------------- helper -----------------------------------------
def _sid(req) -> str:
    """Return existing or new session-id (uuid4)."""
    sid = req.cookies.get(SESSION_COOKIE_KEY)
    if not sid:
        sid = str(uuid.uuid4())
    if sid not in CHAT_LOGS:
        CHAT_LOGS[sid] = []          # initialise chat history
    return sid


def _run_rag(prompt: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Wrapper around rag_scipdf_core.smart_query().
    Returns:
      html_answer  – safe HTML (markdown → html)
      media_list   – [("img", rel_path | url), ("tbl", rel_path | url), …]
    """
    answer_text, media = smart_query(prompt, return_media=True)  # <-- small helper added in rag_scipdf_core
    # answer_text is markdown.  Convert ↓
    html_answer = markdown2.markdown(answer_text, extras=["fenced-code-blocks"])

    # Convert media paths (object_store/…) → url routes /media/…
    show = []
    for kind, p in media:
        p = Path(p).resolve()
        if kind == "img" and OBJ_DIR_IMG in p.parents:
            show.append((kind, f"/media/image/{p.name}"))
        elif kind == "tbl" and OBJ_DIR_TBL in p.parents:
            show.append((kind, f"/media/table/{p.name}"))
    return html_answer, show


# ---------------- routes -----------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json(force=True)
    user_msg = (data or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "empty"}), 400

    sid = _sid(request)

    # 1) store user message
    CHAT_LOGS[sid].append({
        "role": "user",
        "html": markdown2.markdown(user_msg),
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds")
    })

    # 2) call RAG
    try:
        answer_html, media = _run_rag(user_msg)
    except Exception as e:
        answer_html = f"<p style='color:red'>Server error: {e}</p>"
        media = []

    # 3) store assistant answer
    CHAT_LOGS[sid].append({
        "role": "assistant",
        "html": answer_html,
        "media": media,
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds")
    })

    # 4) build payload (latest two messages)
    payload = {
        "answer_html": answer_html,
        "media": media
    }
    resp = make_response(jsonify(payload))
    resp.set_cookie(SESSION_COOKIE_KEY, sid, max_age=60*60*24*30)  # 30 days
    return resp


# expose figures / tables ------------------------------------------------
@app.route("/media/image/<path:filename>")
def media_image(filename):
    return send_from_directory(OBJ_DIR_IMG, filename)

@app.route("/media/table/<path:filename>")
def media_table(filename):
    """Convert markdown file to HTML on the fly."""
    md_path = OBJ_DIR_TBL / filename
    if not md_path.exists():
        return "Not found", 404
    md_text = md_path.read_text(encoding="utf-8")
    html = markdown2.markdown(md_text)
    return f"<html><body>{html}</body></html>"


# recent history (optional sidebar) --------------------------------------
@app.route("/history", methods=["GET"])
def history():
    sid = _sid(request)
    hist = CHAT_LOGS.get(sid, [])
    return jsonify(hist)


# ───────── main ─────────
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

