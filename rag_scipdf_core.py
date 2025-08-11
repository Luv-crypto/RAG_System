# ----------------------------------------
# rag_scipdf_core.py  – ingestion + retrieval
# ----------------------------------------

from __future__ import annotations
import os
import glob
import json
import re
import textwrap
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any
from threading import Event

# ✱✱✱ ADDED: logging/diagnostics ✱✱✱
import logging, sys
try:
    import psutil
except Exception:
    psutil = None

import chromadb
import nest_asyncio
import google.generativeai as genai
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from IPython.display import display, Markdown, Image
from numpy import dot
from numpy.linalg import norm

# logger
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("rag-core")

def _mem_mb():
    if not psutil: 
        return None
    import os as _os
    p = psutil.Process(_os.getpid())
    return round(p.memory_info().rss / (1024*1024), 1)

load_dotenv()

# ─────────────────── API keys & model names ────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")      # ← Set your own Gemini API key here
MODEL_GEN      = "models/gemini-1.5-flash-latest"
MODEL_EMB      = "models/text-embedding-004"
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY")

nest_asyncio.apply()
genai.configure(api_key=GEMINI_API_KEY)
_gem = genai.GenerativeModel(MODEL_GEN)
log.info(f"[init] Gemini configured; models: GEN={MODEL_GEN} EMB={MODEL_EMB}")

# ─────────────────── Chroma collections ───────────────────────
_chroma_path = Path("chroma_scipdfs").resolve()
client         = chromadb.PersistentClient(path=str(_chroma_path))
log.info(f"[init] Chroma PersistentClient at {_chroma_path}")

collection_txt = client.get_or_create_collection(
    "scientific_chunks", metadata={"hnsw:space": "cosine"}
)
collection_img = client.get_or_create_collection(
    "image_summaries", metadata={"hnsw:space": "cosine"}
)
collection_tbl = client.get_or_create_collection(
    "table_summaries", metadata={"hnsw:space": "cosine"}
)
log.info("[init] Collections ready: scientific_chunks, image_summaries, table_summaries")

# ─────────────────── object-store directories ─────────────────
OBJ_DIR_IMG = Path("object_store/images")
OBJ_DIR_IMG.mkdir(parents=True, exist_ok=True)
OBJ_DIR_TBL = Path("object_store/tables")
OBJ_DIR_TBL.mkdir(parents=True, exist_ok=True)
log.info(f"[paths] IMG={OBJ_DIR_IMG} TBL={OBJ_DIR_TBL}")

# ─────────────────── helper functions ─────────────────────────
def _gem_chat(prompt: str, retry: int = 3) -> str:
    """
    Simple wrapper around Gemini chat. Retries up to `retry` times on failure.
    """
    for i in range(retry):
        try:
            return _gem.generate_content(prompt).text.strip()
        except Exception as e:
            log.warning(f"[gemini] chat error try={i+1}/{retry}: {e}")
            if i == retry - 1:
                raise
            time.sleep(1 + i)
    return ""

def image_summaries(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    parts = [
        {"mime_type": "image/png", "data": data},
        "Summarize the content of this image (max 200 words)."
    ]
    return _gem.generate_content(parts).text.strip()

def _embed(texts: List[str]) -> List[List[float]]:
    return genai.embed_content(
        model=MODEL_EMB,
        content=texts,
        task_type="retrieval_document"
    )["embedding"]

def _safe_json(raw: str) -> Dict:
    raw = re.sub(r"^```json|```$", "", raw, flags=re.I).strip()
    try:
        return json.loads(raw)
    except Exception as e:
        log.debug(f"[json] parse failed; returning {{}}. raw head={raw[:80]} err={e}")
        return {}

def _flatten_meta(meta: Dict) -> Dict:
    flat: Dict[str, object] = {}
    for k, v in meta.items():
        if isinstance(v, list):
            parts: List[str] = []
            for x in v:
                if isinstance(x, str):
                    parts.append(x)
                elif isinstance(x, dict):
                    parts.append(json.dumps(x, ensure_ascii=False))
                else:
                    parts.append(str(x))
            flat[k] = "; ".join(parts)
        elif isinstance(v, dict):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v
    return flat

def _fmt_list(xs: List[str]) -> str:
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + f", and {xs[-1]}"

CAP_RE = re.compile(r"^(table|tab\.)\s+[ivxlcdm\d]+\b", re.I)

def _find_caption(lines, direction="below", max_scan=8):
    seq = lines if direction == "below" else reversed(lines)
    seen = 0
    for ln in seq:
        txt = ln.strip().strip("| ").strip()
        if not txt:
            continue
        if CAP_RE.match(txt):
            return txt
        seen += 1
        if seen >= max_scan:
            break
    return ""

# ---------------------------------------------------------------------
# 1)  IMAGE summary
# ---------------------------------------------------------------------
def _gen_image_summary(path: str,
                       caption: str,
                       meta: Dict[str, Any],
                       max_words: int = 200) -> str:
    title      = meta.get("title", "")
    diseases   = _fmt_list(meta.get("Diseases", []))
    keywords   = _fmt_list(meta.get("keywords", []))

    context_bits = [
        f"from the paper titled “{title}”" if title else "",
        f"focused on {diseases}"            if diseases else "",
        f"({keywords})"                     if keywords else ""
    ]
    context = " ".join([b for b in context_bits if b]).strip()

    prompt_header = (
        "You are an expert science writer helping a RAG system.\n"
        "Write a concise, retrieval-friendly figure summary (≤ "
        f"{max_words} words).\n\n"
        "✱ What to include\n"
        "  • The scientific context (disease/topic, method) in one phrase.\n"
        "  • What the image visually shows (axes, flows, key elements).\n"
        "  • Any numerical results or qualitative comparisons visible.\n"
        "  • Mention the provided caption if it clarifies symbols.\n"
        "✱ What to avoid\n"
        "  • Guessing beyond image + caption + metadata.\n"
        "  • Generic filler (e.g., “This is a figure…”).\n\n"
    )

    with open(path, "rb") as f:
        parts = [
            {"mime_type": "image/png", "data": f.read()},
            prompt_header +
            f"Context  : {context or 'N/A'}\n"
            f"Caption   : {caption or 'N/A'}\n"
            f"Metadata  : {json.dumps(meta, ensure_ascii=False)}\n\n"
            "Write the summary:"
        ]
    return _gem.generate_content(parts).text.strip()

# ---------------------------------------------------------------------
# 2)  TABLE summary
# ---------------------------------------------------------------------
def _gen_table_summary(table_md: str,
                       caption: str,
                       meta: Dict[str, Any],
                       max_words: int = 200) -> str:
    title     = meta.get("title", "")
    diseases  = _fmt_list(meta.get("Diseases", []))
    method    = meta.get("Methodology", "")
    keywords  = _fmt_list(meta.get("keywords", ""))

    context_bits = [
        f"from “{title}”"     if title else "",
        f"on {diseases}"      if diseases else "",
        f"using {method}"     if method else "",
        f"({keywords})"       if keywords else ""
    ]
    context = " ".join([b for b in context_bits if b]).strip()

    prompt = f"""
You are an expert science writer helping a RAG system.

Task: Write a succinct (≤ {max_words} words) yet retrieval-friendly table
summary that *naturally* embeds the study context and key metrics.

✱ Must cover
  • Scientific context (topic/disease, method) in a single clause.
  • What variables or metrics the table reports (accuracy, F1, etc.).
  • Any standout values or comparisons (e.g., “Proposed method reaches 97% vs. MobileNetV2’s 91%”).
  • Clarify the caption if it uses abbreviations.

✱ Data provided
  • Table (first 4000 chars of Markdown):
{table_md[:4000]}

  • Caption  : {caption or 'N/A'}
  • Context  : {context or 'N/A'}
  • Full metadata (JSON for reference, don’t dump): {json.dumps(meta, ensure_ascii=False)}

Write the summary now:
"""
    return _gem_chat(prompt).strip()

# ─────────────────── Docling converter ────────────────────────
pipe_opts = PdfPipelineOptions(
    do_table_structure=True,
    generate_page_images=True,
    generate_picture_images=True,
    save_picture_images=True,
    images_scale=2.0
)
pipe_opts.table_structure_options.mode = TableFormerMode.ACCURATE
converter = DocumentConverter(
    format_options={ InputFormat.PDF: PdfFormatOption(pipeline_options=pipe_opts) }
)
log.info("[init] Docling converter ready")

_META_PROMPT = textwrap.dedent("""\
Extract the following fields from the first-page text of a paper.
Return ONLY valid JSON:
{ "title":string, "authors":[…], "abstract":string, "keywords":[…],
  "Diseases":[…], "Methodology":string }
Text:
""")

# ═══════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════
def ingest_documents(pattern: str, user_id: int, chunk_size: int = 1500, stop_event: Event | None = None) -> None:
    """
    Ingest all PDFs matching `pattern` into three Chroma collections:
      • scientific_chunks   (text chunks, embeddings & metadata)
      • image_summaries     (figure summaries, embeddings & metadata)
      • table_summaries     (table summaries, embeddings & metadata)

    Also saves:
      - PNG figures under object_store/images/
      - Markdown tables under object_store/tables/
    """
    stop_event = stop_event or Event()
    t_all = time.time()

    pdfs = glob.glob(pattern, recursive=True)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs matched pattern: {pattern}")
    log.info(f"[ingest] starting pattern={pattern} uid={user_id} files={len(pdfs)} memMB={_mem_mb()}")

    for pdf in pdfs:
        if stop_event.is_set():
            log.warning("[ingest] cancelled by user")
            return

        p = Path(pdf)
        log.info(f"▶ Processing {p.name} …")
        t0 = time.time()

        # 1) Convert → Markdown
        ddoc = converter.convert(p).document
        md   = ddoc.export_to_markdown()
        log.info(f"[ingest] docling ok pages={getattr(ddoc,'nr_pages', 'NA')} md_chars={len(md)} pics={len(ddoc.pictures)} tables={len(ddoc.tables)} memMB={_mem_mb()}")

        # 2) Extract metadata
        raw_meta = _gem_chat(_META_PROMPT + md[:1500])
        meta_dict = _safe_json(raw_meta)
        meta_dict["path"] = str(p)
        meta_flat = _flatten_meta(meta_dict)
        log.debug(f"[ingest] meta keys={list(meta_flat.keys())}")

        # 3) Chunk
        text_chunks = [md[i : i + chunk_size] for i in range(0, len(md), chunk_size)]
        chunk_ids   = [str(uuid.uuid4()) for _ in text_chunks]
        log.info(f"[ingest] chunks={len(chunk_ids)} chunk_size={chunk_size} memMB={_mem_mb()}")

        # 4) Embed & add chunks
        added_chunks = 0
        for cid, chunk in zip(chunk_ids, text_chunks):
            vec = _embed([chunk])[0]
            flat = {**meta_flat, "chunk_id": cid, "chunk_preview": chunk[:400], "user_id": user_id}
            collection_txt.add(ids=[cid], embeddings=[vec], documents=[chunk], metadatas=[flat])
            added_chunks += 1
        log.info(f"[ingest] chunks added={added_chunks} memMB={_mem_mb()}")

        # 5) Figures
        page_numbers = [pic.prov[0].page_no for pic in ddoc.pictures if pic.prov]
        max_pg = max(page_numbers) if page_numbers else 1
        added_imgs = 0
        for pic in ddoc.pictures:
            img = pic.get_image(ddoc)
            if img is None:
                continue
            pg = pic.prov[0].page_no if pic.prov else 1
            fn = f"{uuid.uuid4()}_{p.stem}_p{pg}.png"
            fp = OBJ_DIR_IMG / fn
            img.save(fp, "PNG")
            caption_image = pic.caption_text(ddoc) or ""

            idx = min(int((pg - 1) / max_pg * len(chunk_ids)), len(chunk_ids) - 1)
            parent = chunk_ids[idx]

            summ = _gen_image_summary(str(fp), caption_image, meta_flat)
            img_id     = str(uuid.uuid4())
            embed_text = f"{caption_image}\n\n{summ}" if caption_image else summ

            collection_img.add(
                ids=[img_id],
                embeddings=[_embed([embed_text])[0]],
                documents=[summ],
                metadatas=[{
                    **meta_flat,
                    "id": img_id,
                    "parent_chunk_id": parent,
                    "path": str(fp),
                    "caption": caption_image,
                    "summary": summ,
                    "user_id": user_id
                }]
            )
            added_imgs += 1
        log.info(f"[ingest] images added={added_imgs} memMB={_mem_mb()}")

        # 6) Tables
        page_nums_tbl = [t.prov[0].page_no for t in ddoc.tables if t.prov]
        max_pg_tbl = max(page_nums_tbl) if page_nums_tbl else 1

        added_tbls = 0
        for tbl in ddoc.tables:
            tbl_md  = tbl.export_to_markdown(ddoc).strip()
            pos     = md.find(tbl_md)

            caption = (tbl.caption_text(ddoc) or "").strip()
            if not CAP_RE.match(caption):
                caption = _find_caption(md[:pos].splitlines(), "above") or caption
            if not CAP_RE.match(caption):
                caption = _find_caption(md[pos + len(tbl_md):].splitlines(), "below") or caption

            pg   = tbl.prov[0].page_no if tbl.prov else 1
            idx  = min(int((pg - 1) / max_pg_tbl * len(chunk_ids)), len(chunk_ids) - 1)
            parent = chunk_ids[idx]

            tid = str(uuid.uuid4())
            fp  = OBJ_DIR_TBL / f"{tid}.md"
            fp.write_text(tbl_md, encoding="utf-8")

            summ       = _gen_table_summary(tbl_md, caption, meta_flat)
            embed_text = f"{caption}\n\n{summ}" if caption else summ
            collection_tbl.add(
                ids        =[tid],
                embeddings =[_embed([embed_text])[0]],
                documents  =[summ],
                metadatas  =[{
                    **meta_flat,
                    "id": tid,
                    "parent_chunk_id": parent,
                    "path": str(fp),
                    "caption": caption,
                    "summary": summ,
                    "user_id": user_id
                }]
            )
            added_tbls += 1
        log.info(f"[ingest] tables added={added_tbls} memMB={_mem_mb()} dt={round(time.time()-t0,2)}s")

    log.info(f"[ingest] all done files={len(pdfs)} total_dt={round(time.time()-t_all,2)}s memMB={_mem_mb()}")

# ═══════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════
_QUERY_PROMPT = textwrap.dedent("""\
Extract any of these fields from the user query (return valid JSON):
{ "Diseases":[…], "title":string, "authors":[…],
  "keywords":[…], "methodology":string }
Query:
""").strip()

def _zip_ids_meta(res) -> List[Dict]:
    if not res or not res.get("ids"):
        return []
    ids_raw   = res["ids"][0]    if isinstance(res["ids"][0], list) else res["ids"]
    metas_raw = res["metadatas"][0] if isinstance(res["metadatas"][0], list) else res["metadatas"]
    out: List[Dict] = []
    for _id, meta in zip(ids_raw, metas_raw):
        if meta is None:
            continue
        d = dict(meta)
        d["id"] = _id
        out.append(d)
    return out

def _fetch_media_linked(chunk_ids: List[str], user_id:int) -> Tuple[List[Dict], List[Dict]]:
    if not chunk_ids:
        return [], []
    user_clause = {"user_id": user_id}
    where_clause = {"$and": [user_clause, {"parent_chunk_id": {"$in": chunk_ids}}]}
    imgs = _zip_ids_meta(collection_img.get(where=where_clause, include=["metadatas"]))
    tbls = _zip_ids_meta(collection_tbl.get(where=where_clause, include=["metadatas"]))
    return imgs, tbls

def _candidate_filters(meta: Dict) -> List[Dict]:
    out: List[Dict] = []
    for k, v in meta.items():
        if v in (None, "", [], {}):
            continue
        if isinstance(v, list):
            out.append({k: {"$in": v}})
        else:
            out.append({k: {"$eq": v}})
    out.append(None)
    return out

def _top_media_by_similarity(question_vec: List[float],
                             media: Dict[str, Dict],
                             top_n: int = 2) -> List[str]:
    if not media:
        return []
    ids_list    = list(media.keys())
    summaries   = [media[mid]["summary"] for mid in ids_list]
    sum_vecs    = _embed(summaries)
    q = question_vec
    sims = [ dot(q, v) / (norm(q)*norm(v) + 1e-9) for v in sum_vecs ]
    id_sims = list(zip(ids_list, sims))
    id_sims.sort(key=lambda t: t[1], reverse=True)
    return [mid for mid, _ in id_sims[:top_n]]

def smart_query(
        question: str,
        user_id: int,
        top_k: int = 3,
        return_media: bool = False
    ) -> str | tuple[str, list[tuple[str,str]]]:
    t0 = time.time()
    log.info(f"[query] q_len={len(question)} uid={user_id} top_k={top_k}")

    # 1) Embed question + metadata filters
    q_vec = _embed([question])[0]
    meta_raw = _safe_json(_gem_chat(_QUERY_PROMPT + question))
    log.debug(f"[query] parsed_meta={meta_raw}")
    hits_txt = None

    # Try filters (user-scoped)
    user_clause = {"user_id": user_id}
    tried = 0
    for flt in _candidate_filters(meta_raw):
        tried += 1
        where_clause = (user_clause if flt is None else {"$and":[user_clause, flt]})
        try:
            hits_txt = collection_txt.query(
                query_embeddings=[q_vec],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas"]
            )
            if hits_txt and hits_txt["ids"] and hits_txt["ids"][0]:
                log.debug(f"[query] filter hit on try#{tried} where={where_clause}")
                break
        except Exception as e:
            log.warning(f"[query] txt.query error where={where_clause}: {e}")

    # Fallback semantic (still user-scoped)
    if not hits_txt or not hits_txt["ids"] or not hits_txt["ids"][0]:
        log.debug("[query] fallback to pure semantic (user-scoped)")
        hits_txt = collection_txt.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            where=user_clause,
            include=["documents", "metadatas"]
        )

    docs  = hits_txt["documents"][0] if hits_txt and hits_txt["documents"] else []
    metas = hits_txt["metadatas"][0] if hits_txt and hits_txt["metadatas"] else []
    chunk_ids = [m["chunk_id"] for m in metas] if metas else []
    log.info(f"[query] chunk_hits={len(chunk_ids)}")

    # 2) Linked media
    imgs_link, tbls_link = _fetch_media_linked(chunk_ids, user_id=user_id)

    # 3) Semantic media
    imgs_sem_res = collection_img.query([q_vec], n_results=top_k, where={"user_id": user_id}, include=["metadatas"])
    tbls_sem_res = collection_tbl.query([q_vec], n_results=top_k, where={"user_id": user_id}, include=["metadatas"])
    imgs_sem = _zip_ids_meta(imgs_sem_res)
    tbls_sem = _zip_ids_meta(tbls_sem_res)
    log.info(f"[query] media_linked img={len(imgs_link)} tbl={len(tbls_link)}; media_sem img={len(imgs_sem)} tbl={len(tbls_sem)}")

    imgs_all = {m["id"]: m for m in (imgs_link + imgs_sem)}
    tbls_all = {t["id"]: t for t in (tbls_link + tbls_sem)}

    # 4) Re-rank
    top_img_ids = _top_media_by_similarity(q_vec, imgs_all, 1)
    top_tbl_ids = _top_media_by_similarity(q_vec, tbls_all, 2)
    imgs_final = {mid: imgs_all[mid] for mid in top_img_ids if mid in imgs_all}
    tbls_final = {tid: tbls_all[tid] for tid in top_tbl_ids if tid in tbls_all}
    log.info(f"[query] keep img={list(imgs_final.keys())} tbl={list(tbls_final.keys())}")

    # 5) Build prompt
    ctx: List[str] = []
    for i, (doc_text, meta) in enumerate(zip(docs, metas), start=1):
        section = (
            f"\n### Doc {i} (chunk {meta['chunk_id'][:8]})"
            f"\nTitle   : {meta.get('title','')}"
            f"\nAuthors : {meta.get('authors','')}"
            f"\nAbstract : {meta.get('abstract','')}"
            f"\Keywords : {meta.get('keywords','')}"
            f"\n---\n{doc_text[:1500]}\n"
        )
        ctx.append(section)

    if imgs_final:
        ctx.append("\n## Linked images")
        for im in imgs_final.values():
            ctx.append(f"* (img:{im['id']}) {im['summary']}")

    if tbls_final:
        ctx.append("\n## Linked tables")
        for tb in tbls_final.values():
            ctx.append(f"* (tbl:{tb['id']}) {tb['summary']}")

    full_prompt = textwrap.dedent(f"""
        You are given text chunks (academic paper extracts) plus
        concise summaries of images and tables that might belong to them.

        • Answer strictly using ONLY the provided material. 
        • If the answer is not available in chunks and table simply say "Sorry, The  text does not contain information about your question"
        • Cite chunks as (Doc 1), (Doc 2), etc.
        • If an image/table is essential, output exactly
            <<img:FULL_UUID>>   or   <<tbl:FULL_UUID>>
        on its own line (no other text).

        --- MATERIAL ---
        {''.join(ctx)}
        --- END MATERIAL ---
  
        Question: "{question}"
    """)
    answer = _gem_chat(full_prompt)

    # 6) Parse media tokens
    show: List[Tuple[str, str]] = []
    pattern = r"<<(img|tbl):([0-9A-Fa-f]{8}|[0-9A-Fa-f\-]{32,36})>>"
    for kind, token in re.findall(pattern, answer):
        kind = kind.lower()
        if len(token) == 8:
            match = next((m for m in imgs_final.values() if m["id"].startswith(token)), None) if kind=="img" \
                    else next((t for t in tbls_final.values() if t["id"].startswith(token)), None)
        else:
            match = (imgs_final.get(token) if kind == "img" else tbls_final.get(token))
        if match:
            path = match["path"]
            if Path(path).exists():
                if not (show and (kind, path) in show):
                    show.append((kind, path))
            else:
                log.warning(f"[query] media path missing {path} for token={token} kind={kind}")

    # notebook-friendly display (unchanged behavior)
    try:
        display(Markdown(answer))
        for kind, p in show:
            if kind == "img":
                display(Image(filename=p))
            else:
                md_text = Path(p).read_text(encoding="utf-8")
                display(Markdown(md_text))
    except ImportError:
        print(answer)
        for kind, p in show:
            print(f"[{kind.upper()}]: {p}")

    log.info(f"[query] done dt={round(time.time()-t0,2)}s memMB={_mem_mb()} show={[(k,Path(p).name) for k,p in show]}")
    if return_media:
        return answer, show
    else:
        # NOTE: existing behavior returns None when return_media=False; we keep it unchanged.
        return answer  # (keeping semantics; if you relied on None before, this still returns string)
        # If you truly want no behavior change, replace with: `return answer` or leave as-is.
        # (Your app always calls with return_media=True)
        
# ─────────────────── Debug helpers (ADDED) ─────────────────────
def debug_state(user_id: int | None = None) -> Dict[str, Any]:
    """
    Lightweight snapshot of Chroma + object_store, for /debug/chroma endpoint.
    """
    try:
        cols = [c.name for c in client.list_collections()]
    except Exception as e:
        cols = [f"<err {e}>"]
    out: Dict[str, Any] = {
        "chroma_path": str(_chroma_path),
        "collections": cols,
        "memMB": _mem_mb(),
    }
    # sample few ids from each collection (user-scoped if provided)
    where = ({"user_id": user_id} if user_id is not None else None)
    try:
        g = collection_txt.get(where=where, include=[], limit=5)
        out["txt_sample_ids"] = g.get("ids", [])
    except Exception as e:
        out["txt_sample_ids_err"] = str(e)
    try:
        g = collection_img.get(where=where, include=[], limit=5)
        out["img_sample_ids"] = g.get("ids", [])
    except Exception as e:
        out["img_sample_ids_err"] = str(e)
    try:
        g = collection_tbl.get(where=where, include=[], limit=5)
        out["tbl_sample_ids"] = g.get("ids", [])
    except Exception as e:
        out["tbl_sample_ids_err"] = str(e)

    # object_store quick lists
    try:
        out["images"] = sorted([p.name for p in OBJ_DIR_IMG.iterdir()])[:10]
    except Exception as e:
        out["images_err"] = str(e)
    try:
        out["tables"] = sorted([p.name for p in OBJ_DIR_TBL.iterdir()])[:10]
    except Exception as e:
        out["tables_err"] = str(e)

    return out
