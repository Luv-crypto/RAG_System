import os, uuid, datetime, json, markdown2
from pathlib import Path
from typing import List, Tuple
import threading, secrets, queue
import sqlite3, hashlib, secrets

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, make_response ,redirect, url_for, Response
    )

# ✱✱✱ ADDED: logging + diagnostics (no behavior change) ✱✱✱
import logging, sys, time, shutil
try:
    import psutil  # present in your image; if not, we just skip mem logs
except Exception:
    psutil = None

# ---------- your RAG core (imported) ---------------------------
from rag_scipdf_core import smart_query   # <- must be importable!
from dotenv import load_dotenv

# app.py
os.environ["ANONYMIZED_TELEMETRY"] = os.getenv("ANONYMIZED_TELEMETRY", "False")
app = Flask(__name__)

# ✱✱✱ ADDED: logger config ✱✱✱
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("rag-app")

# Load .env into process environment
load_dotenv()

# ---------- constants ------------------------------------------
ROOT               = Path(__file__).parent.resolve()
OBJ_DIR_IMG        = ROOT / "object_store" / "images"
OBJ_DIR_TBL        = ROOT / "object_store" / "tables"
SESSION_COOKIE_KEY = "sid"

# ---------- Flask -------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")

# ---------- in-memory session store ------------------------------
# { session_id : [ {role, html, ts}, ... ] }
CHAT_LOGS = {}
INGEST_TASKS = {}

# ✱✱✱ ADDED: startup snapshot ✱✱✱
def _mem_mb():
    if not psutil: 
        return None
    p = psutil.Process()
    return round(p.memory_info().rss / (1024*1024), 1)

def _disk_info(p: Path):
    try:
        u = shutil.disk_usage(p)
        return {"total_GB": round(u.total/1e9,2), "used_GB": round((u.total-u.free)/1e9,2), "free_GB": round(u.free/1e9,2)}
    except Exception as e:
        return {"err": str(e)}

log.info("===== RAG web startup =====")
log.info(f"LOG_LEVEL={LOG_LEVEL}")
log.info(f"ROOT={ROOT}")
log.info(f"OBJ_DIR_IMG={OBJ_DIR_IMG} exists={OBJ_DIR_IMG.exists()}")
log.info(f"OBJ_DIR_TBL={OBJ_DIR_TBL} exists={OBJ_DIR_TBL.exists()}")
log.info(f"DB path={(ROOT / 'users.db')}")
log.info(f"Mem RSS MB={_mem_mb()}")
log.info(f"Disk at /app: {_disk_info(ROOT)}")

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
    t0 = time.time()
    uid = _current_uid(request)
    log.info(f"[chat] RAG start uid={uid} msg_len={len(prompt)}")
    answer_text, media = smart_query(prompt, user_id= uid , return_media=True)  # <-- small helper added in rag_scipdf_core
    dt = round(time.time()-t0,2)
    log.info(f"[chat] RAG done in {dt}s; media={[(k,Path(p).name) for k,p in media]}")
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
    log.debug(f"[chat] key={key} len={len(user_msg)}")

    # 1) store user msg
    uid = _current_uid(request)
    db = _get_db()
    db.execute("INSERT INTO chats (user_id,role,html,ts) VALUES (?,?,?,?)",
            (uid, "user", markdown2.markdown(user_msg), datetime.datetime.utcnow().isoformat(timespec="seconds")
))
    db.commit()

    # 2) call RAG
    try:
        answer_html, media = _run_rag(user_msg)
    except Exception as e:
        log.exception("[chat] RAG error")
        answer_html = f"<p style='color:red'>Server error: {e}</p>"
        media = []

    # 3) build media tags
    media_html = []
    for kind, url in media:
        if kind == "img":
            media_html.append(f'<img src="{url}" class="inline-img">')
        else:  # tables
            media_html.append(f'<iframe src="{url}" class="tbl-frame"></iframe>')

    # 4) store assistant answer
    db.execute("INSERT INTO chats (user_id,role,html,ts) VALUES (?,?,?,?)",
           (uid, "assistant", answer_html + "".join(media_html),
           datetime.datetime.utcnow().isoformat(timespec="seconds") ))
    db.commit()
    log.debug(f"[chat] stored response; uid={uid} media_count={len(media)}")

    return jsonify({
        "answer_html": answer_html,
        "media": media
    })

# expose figures / tables ------------------------------------------------
@app.route("/media/image/<path:filename>")
def media_image(filename):
    fp = OBJ_DIR_IMG / filename
    log.debug(f"[media] image {fp} exists={fp.exists()} size={fp.stat().st_size if fp.exists() else 'NA'}")
    return send_from_directory(OBJ_DIR_IMG, filename)

@app.route("/media/table/<path:filename>")
def media_table(filename):
    md_path = OBJ_DIR_TBL / filename
    if not md_path.exists():
        log.warning(f"[media] table missing {md_path}")
        return "Not found", 404

    md_text = md_path.read_text(encoding="utf-8")
    html = markdown2.markdown(md_text, extras=["tables"])

    css = """
      <style>
        table{border-collapse:collapse;width:100%;}
        th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}
      </style>
    """
    return f"<html><head>{css}</head><body>{html}</body></html>"

# recent history ---------------------------------------------------------
@app.route("/history")
def history():
    uid = _current_uid(request)
    if uid is None:
        return jsonify([])
    rows = _get_db().execute(
        "SELECT role,html,ts FROM chats WHERE user_id=? ORDER BY id", (uid,)
    ).fetchall()
    return jsonify([{"role":r[0],"html":r[1],"ts":r[2]} for r in rows])

# ───────────────────────────────────────────────────────────────
# 0)  Imports & database helpers
# ───────────────────────────────────────────────────────────────
DB_PATH = ROOT / "users.db"

def _get_db():
    db = sqlite3.connect(DB_PATH)
    # ensure tables exist
    db.execute("""
      CREATE TABLE IF NOT EXISTS users (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT UNIQUE,
        pw_hash   TEXT
      )
    """)
    db.execute("""
      CREATE TABLE IF NOT EXISTS chats (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   INTEGER,
        role      TEXT,    -- 'user' or 'assistant'
        html      TEXT,
        ts        TEXT
      )
    """)
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
        log.info(f"[auth] registered user={user} id={uid}")
        resp = redirect(url_for("index"))
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
            log.warning(f"[auth] bad login for user={user}")
            return "Bad credentials", 401
        log.info(f"[auth] login ok user={user} id={row[0]}")
        resp = redirect(url_for("index"))
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
PUBLIC_PATHS = {"/login", "/register", "/static/", "/media/", "/health", "/debug/env", "/debug/paths", "/debug/chroma"}

@app.before_request
def force_login():
    # Let Flask serve static files & auth pages without a login
    path = request.path
    if any(path.startswith(p) for p in PUBLIC_PATHS):
        return
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

    if request.method == "GET":
        return render_template("upload.html")

    f = request.files.get("pdf")
    if not (f and f.filename.lower().endswith(".pdf")):
        return "PDF only", 400

    user_dir = UPLOAD_DIR / f"user_{uid}"
    user_dir.mkdir(exist_ok=True)

    save_path = user_dir / f"{secrets.token_hex(8)}_{f.filename}"
    f.save(save_path)
    log.info(f"[upload] uid={uid} saved={save_path} size={save_path.stat().st_size}")
    return jsonify({"file_path": str(save_path)}), 200

@app.route("/ingest", methods=["POST"])
def ingest():
    """Kick off ingestion in a background thread, return task_id."""
    uid = _current_uid(request)
    data = request.get_json(force=True)
    path = Path(data.get("file_path", ""))
    if not path.exists():
        log.warning(f"[ingest] file not found: {path}")
        return "file not found", 400

    stop_flag = threading.Event()
    task_id   = secrets.token_hex(8)
    log.info(f"[ingest] start task={task_id} uid={uid} file={path}")

    def worker():
        from rag_scipdf_core import ingest_documents
        t0 = time.time()
        try:
            ingest_documents(str(path), stop_event=stop_flag, user_id=uid)
            INGEST_TASKS[task_id]["status"] = "complete"
            log.info(f"[ingest] done task={task_id} dt={round(time.time()-t0,2)}s memMB={_mem_mb()}")
        except Exception as e:
            INGEST_TASKS[task_id]["status"] = f"failed: {e}"
            log.exception(f"[ingest] failed task={task_id}")
        finally:
            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    INGEST_TASKS[task_id] = {"thread": t, "stop": stop_flag,"status":"running"}
    return jsonify({"task_id": task_id}), 202

@app.route("/ingest/cancel/<task_id>", methods=["POST"])
def cancel_ingest(task_id):
    task = INGEST_TASKS.get(task_id)
    if not task: return "task not found", 404
    task["stop"].set()
    task["status"] = "cancelled"
    log.info(f"[ingest] cancelled task={task_id}")
    return "cancelled", 200

@app.route("/ingest/status/<task_id>")
def ingest_status(task_id):
    task = INGEST_TASKS.get(task_id)
    if not task:
        return jsonify({"status": "unknown"}), 404
    return jsonify({"status": task.get("status", "running")})

# ───────── health / debug (ADDED, optional) ─────────
@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/debug/env")
def debug_env():
    keys = ["FLASK_RUN_HOST","FLASK_RUN_PORT","FLASK_DEBUG","LOG_LEVEL","GEMINI_API_KEY","ANONYMIZED_TELEMETRY"]
    safe = {k: ("<set>" if os.getenv(k) else "") for k in keys}
    safe["memMB"] = _mem_mb()
    safe["disk_app"] = _disk_info(ROOT)
    return jsonify(safe)

@app.route("/debug/paths")
def debug_paths():
    def _ls(p: Path):
        try:
            return sorted([f.name for f in p.iterdir()])
        except Exception as e:
            return [f"<err {e}>"]
    return jsonify({
        "OBJ_DIR_IMG": str(OBJ_DIR_IMG),
        "OBJ_DIR_TBL": str(OBJ_DIR_TBL),
        "IMG_files": _ls(OBJ_DIR_IMG) if OBJ_DIR_IMG.exists() else [],
        "TBL_files": _ls(OBJ_DIR_TBL) if OBJ_DIR_TBL.exists() else [],
        "DB_exists": (ROOT/"users.db").exists(),
    })

@app.route("/debug/chroma")
def debug_chroma():
    try:
        from rag_scipdf_core import debug_state  # ADDED helper in core
        uid = _current_uid(request) or -1
        return jsonify(debug_state(uid))
    except Exception as e:
        log.exception("[debug] chroma state error")
        return jsonify({"error": str(e)}), 500

# ───────── main ─────────
def main():
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("1","true","yes")
    log.info(f"Flask running on {host}:{port} debug={debug}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)

if __name__ == "__main__":
    main()
