

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


# ---------------- helper -----------------------------------------j
# ---------------- helper -----------------------------------------
def _chat_key(req) -> str:
    """
    Return the key under which this user's chat history is stored.
    Uses uid cookie; if somehow anonymous, falls back to a per-browser UUID.
    """
    uid = _current_uid(req)
    if uid is not None:                     # logged-in user
        return f"user_{uid}"                # e.g. "user_5"

    # ------- anonymous (should not happen after login guard) -------
    anon = req.cookies.get("sid")
    if not anon:
        anon = str(uuid.uuid4())
    return f"anon_{anon}"



def _run_rag(prompt: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Wrapper around rag_scipdf_core.smart_query().
    Returns:
      html_answer  – safe HTML (markdown → html)
      media_list   – [("img", rel_path | url), ("tbl", rel_path | url), …]
    """
    answer_text, media = smart_query(prompt, return_media=True)  # <-- small helper added in rag_scipdf_core
    # answer_text is markdown.  Convert ↓
    html_answer = markdown2.markdown(answer_text, extras=["fenced-code-blocks", "tables"])

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

    key = _chat_key(request)

# 1) store user message
    CHAT_LOGS.setdefault(key, []).append({
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
    # 3) build <img> / <iframe> tags once so history can replay them
    media_html = []
    for kind, url in media:
        if kind == "img":
            media_html.append(f'<img src="{url}" class="inline-img">')
        else:  # tables
            media_html.append(f'<iframe src="{url}" class="tbl-frame"></iframe>')

    CHAT_LOGS[key].append({
    "role": "assistant",
    "html": answer_html + "".join(media_html),
    "ts": datetime.datetime.utcnow().isoformat(timespec="seconds")
        })

    return jsonify({
        "answer_html": answer_html,
        "media": media
    })





# expose figures / tables ------------------------------------------------
@app.route("/media/image/<path:filename>")
def media_image(filename):
    return send_from_directory(OBJ_DIR_IMG, filename)

@app.route("/media/table/<path:filename>")
def media_table(filename):
    md_path = OBJ_DIR_TBL / filename
    if not md_path.exists():
        return "Not found", 404

    md_text = md_path.read_text(encoding="utf-8")
    html = markdown2.markdown(md_text, extras=["tables"])

    # embed minimal CSS so the table is readable even inside the iframe
    css = """
      <style>
        table{border-collapse:collapse;width:100%;}
        th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}
      </style>
    """
    return f"<html><head>{css}</head><body>{html}</body></html>"



# recent history (optional sidebar) --------------------------------------
@app.route("/history", methods=["GET"])
def history():
    sid = _chat_key(request)
    hist = CHAT_LOGS.get(sid, [])
    return jsonify(hist)


# ───────────────────────────────────────────────────────────────
# 0)  Imports & database helpers
# ───────────────────────────────────────────────────────────────
import sqlite3, hashlib, secrets
DB_PATH = ROOT / "users.db"

def _get_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        pw_hash TEXT
      )""")
    return db

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ───────────────────────────────────────────────────────────────
# 1)  User-session helpers
# ───────────────────────────────────────────────────────────────
USER_COOKIE = "uid"

def _current_uid(req) -> int | None:
    try:
        return int(req.cookies.get(USER_COOKIE))
    except (TypeError, ValueError):
        return None
    

def _login_resp(uid: int, resp):
    resp.set_cookie(USER_COOKIE, str(uid), max_age=60*60*24*30, httponly=True)
    return resp

# ───────────────────────────────────────────────────────────────
# 2)  Auth routes
# ───────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user = request.form["user"].strip()
        pw   = request.form["pw"]
        db   = _get_db()
        try:
            db.execute("INSERT INTO users (username,pw_hash) VALUES (?,?)",
                       (user, _hash_pw(pw)))
            db.commit()
        except sqlite3.IntegrityError:
            return "Username taken", 400
        uid = db.execute("SELECT id FROM users WHERE username=?", (user,)).fetchone()[0]
        resp = redirect(url_for("index"))          # 302 → /
        return _login_resp(uid, resp)
    return render_template("login.html", mode="register")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["user"].strip()
        pw   = request.form["pw"]
        db   = _get_db()
        row  = db.execute("SELECT id,pw_hash FROM users WHERE username=?", (user,)).fetchone()
        if not row or _hash_pw(pw) != row[1]:
            return "Bad credentials", 401
        resp = redirect(url_for("index"))          # 302 → /
        return _login_resp(row[0], resp)
        
    return render_template("login.html", mode="login")

@app.route("/logout")
def logout():
    resp = redirect(url_for("login"))
    resp.delete_cookie(USER_COOKIE)
    resp.delete_cookie(SESSION_COOKIE_KEY)
    return resp

from flask import redirect, url_for

# ───────────────────────────────────────────────────────────────
#  GLOBAL LOGIN REQUIRED (except a few routes)
# ───────────────────────────────────────────────────────────────
PUBLIC_PATHS = {"/login", "/register", "/static/", "/media/"}

@app.before_request
def force_login():
    # Let Flask serve static files & auth pages without a login
    path = request.path
    if any(path.startswith(p) for p in PUBLIC_PATHS):
        return

    # Otherwise require a valid user cookie
    if _current_uid(request) is None:
        return redirect(url_for("login"))


# ───────────────────────────────────────────────────────────────
# 3)  PDF-upload route
# ───────────────────────────────────────────────────────────────
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    uid = _current_uid(request)
    if uid is None:
        return "Login first", 401

    if request.method == "POST":
        f = request.files.get("pdf")
        if not (f and f.filename.lower().endswith(".pdf")):
            return "PDF only", 400

        user_folder = UPLOAD_DIR / f"user_{uid}"
        user_folder.mkdir(exist_ok=True)

        save_path = user_folder / f"{secrets.token_hex(8)}_{f.filename}"
        f.save(save_path)

        # call your existing ingestion (per-user collection name)
        from rag_scipdf_core import ingest_documents
        ingest_documents(str(save_path))      # one-file pattern
        return "Ingested!", 200

    return render_template("upload.html")



# ───────── main ─────────
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000,
            debug=True,
            use_reloader=False)   # ← add this


