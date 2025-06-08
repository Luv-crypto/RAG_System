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
from typing import Dict, List, Tuple

import chromadb
import nest_asyncio
import google.generativeai as genai

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# ─────────────────── API keys & model names ────────────────────
GEMINI_API_KEY = "AIzaSyD-zaE26w-qsu00AyXXjVI7HK7Uhl6-IVQ"      # ← Set your own Gemini API key here
MODEL_GEN      = "models/gemini-1.5-flash-latest"
MODEL_EMB      = "models/text-embedding-004"
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY")

nest_asyncio.apply()
genai.configure(api_key=GEMINI_API_KEY)
_gem = genai.GenerativeModel(MODEL_GEN)

# ─────────────────── Chroma collections ───────────────────────
client         = chromadb.PersistentClient(path="chroma_scipdfs")
collection_txt = client.get_or_create_collection(
    "scientific_chunks", metadata={"hnsw:space": "cosine"}
)
collection_img = client.get_or_create_collection(
    "image_summaries", metadata={"hnsw:space": "cosine"}
)
collection_tbl = client.get_or_create_collection(
    "table_summaries", metadata={"hnsw:space": "cosine"}
)

# ─────────────────── object-store directories ─────────────────
OBJ_DIR_IMG = Path("object_store/images")
OBJ_DIR_IMG.mkdir(parents=True, exist_ok=True)
OBJ_DIR_TBL = Path("object_store/tables")
OBJ_DIR_TBL.mkdir(parents=True, exist_ok=True)

# ─────────────────── helper functions ─────────────────────────
def _gem_chat(prompt: str, retry: int = 3) -> str:
    """
    Simple wrapper around Gemini chat. Retries up to `retry` times on failure.
    """
    for i in range(retry):
        try:
            return _gem.generate_content(prompt).text.strip()
        except Exception:
            if i == retry - 1:
                raise
            time.sleep(1 + i)
    # should never reach here
    return ""

def image_summaries(path: str) -> str:
    """
    Given a local PNG `path`, send binary to Gemini and ask for a 200-word summary.
    Returns the summary string.
    """
    with open(path, "rb") as f:
        data = f.read()
    parts = [
        {"mime_type": "image/png", "data": data},
        "Summarize the content of this image (max 200 words)."
    ]
    return _gem.generate_content(parts).text.strip()

def _embed(texts: List[str]) -> List[List[float]]:
    """
    Given a list of strings, return a list of embeddings via Gemini.
    """
    return genai.embed_content(
        model=MODEL_EMB,
        content=texts,
        task_type="retrieval_document"
    )["embedding"]

def _safe_json(raw: str) -> Dict:
    """
    Strip any ```json fences and attempt to json.loads. On failure, return {}.
    """
    raw = re.sub(r"^```json|```$", "", raw, flags=re.I).strip()
    try:
        return json.loads(raw)
    except:
        return {}

def _flatten_meta(meta: Dict) -> Dict:
    """
    Convert a metadata dict so that every value is a scalar (string/int/float/bool).
    - Lists become semicolon-joined JSON dumps of each element (if not a simple string).
    - Dicts become JSON dumps.
    """
    flat: Dict[str, object] = {}
    for k, v in meta.items():
        if isinstance(v, list):
            parts: List[str] = []
            for x in v:
                if isinstance(x, str):
                    parts.append(x)
                elif isinstance(x, dict):
                    # turn nested dict→ JSON string
                    parts.append(json.dumps(x, ensure_ascii=False))
                else:
                    parts.append(str(x))
            flat[k] = "; ".join(parts)
        elif isinstance(v, dict):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v
    return flat

def _summarise(text: str, kind: str) -> str:
    """
    Ask Gemini for a 200-word summary of the given text snippet (first ~5000 chars).
    """
    prompt = f"Give a precise 200-word summary of the following {kind}:\n\n{text[:5000]}"
    return _gem_chat(prompt)

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
def ingest_documents(pattern: str, chunk_size: int = 1500) -> None:
    """
    Ingest all PDFs matching `pattern` into three Chroma collections:
      • scientific_chunks   (text chunks, embeddings & metadata)
      • image_summaries     (figure summaries, embeddings & metadata)
      • table_summaries     (table summaries, embeddings & metadata)

    Also saves:
      - PNG figures under object_store/images/
      - Markdown tables under object_store/tables/
    """
    pdfs = glob.glob(pattern, recursive=True)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs matched pattern: {pattern}")

    # Regex to catch Markdown‐style tables
    tbl_re = re.compile(r"(?:\|.*\n\|[ \t]*[-:]+.*\n(?:\|.*\n)*)", re.MULTILINE)

    for pdf in pdfs:
        p = Path(pdf)
        print(f"\n▶ Processing {p.name} …")
        # 1) Use Docling to convert → Markdown + page images + saved PNGs
        ddoc = converter.convert(p).document
        md   = ddoc.export_to_markdown()

        # 2) Extract “global” metadata (title/authors/etc) from first ~1500 chars
        raw_meta = _gem_chat(_META_PROMPT + md[:1500])
        meta_dict = _safe_json(raw_meta)
        meta_dict["path"] = str(p)
        meta_flat = _flatten_meta(meta_dict)

        # 3) Split the full Markdown into ~chunk_size pieces
        text_chunks = [md[i : i + chunk_size] for i in range(0, len(md), chunk_size)]
        chunk_ids   = [str(uuid.uuid4()) for _ in text_chunks]

        # 4) Embed & store each text chunk into `collection_txt`
        for cid, chunk in zip(chunk_ids, text_chunks):
            vec = _embed([chunk])[0]
            flat = {
                **meta_flat,
                "chunk_id": cid,
                "chunk_preview": chunk[:400]
            }
            collection_txt.add(
                ids=[cid],
                embeddings=[vec],
                documents=[chunk],
                metadatas=[flat]
            )

        # 5) Process each figure in ddoc.pictures → save PNG + embed its 200-word summary
        page_numbers = [pic.prov[0].page_no for pic in ddoc.pictures if pic.prov]
        max_pg = max(page_numbers) if page_numbers else 1
        for pic in ddoc.pictures:
            img = pic.get_image(ddoc)
            if img is None:
                continue
            pg = pic.prov[0].page_no if pic.prov else 1
            # Save PNG to object_store/images/
            fn = f"{uuid.uuid4()}_{p.stem}_p{pg}.png"
            fp = OBJ_DIR_IMG / fn
            img.save(fp, "PNG")

            # Determine which text‐chunk “owns” this page:
            idx = min(int((pg - 1) / max_pg * len(chunk_ids)), len(chunk_ids) - 1)
            parent = chunk_ids[idx]

            # Summarize that figure (send PNG → Gemini)
            summ = image_summaries(str(fp))
            img_id = str(uuid.uuid4())

            collection_img.add(
                ids=[img_id],
                embeddings=[_embed([summ])[0]],
                documents=[summ],
                metadatas=[{
                    **meta_flat,
                    "id": img_id,
                    "parent_chunk_id": parent,
                    "path": str(fp),
                    "summary": summ
                }]
            )

        # 6) Extract every Markdown table → save as `.md` + embed 200-word summary
        for m in tbl_re.finditer(md):
            tbl_md = m.group(0).strip()
            pos    = m.start()
            idx    = pos // chunk_size
            parent = chunk_ids[min(idx, len(chunk_ids) - 1)]

            tid = str(uuid.uuid4())
            fp  = OBJ_DIR_TBL / f"{tid}.md"
            fp.write_text(tbl_md, encoding="utf-8")

            summ = _summarise(tbl_md, "table")
            collection_tbl.add(
                ids=[tid],
                embeddings=[_embed([summ])[0]],
                documents=[summ],
                metadatas=[{
                    **meta_flat,
                    "id": tid,
                    "parent_chunk_id": parent,
                    "path": str(fp),
                    "summary": summ
                }]
            )

        print("✓ Indexed", p.name, f": {len(text_chunks)} chunks + media mapped.")

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
    """
    Convert a Chroma `get()` or `query()` result into a list of dicts,
    each containing the metadata plus an “id” field. If no hits, return [].
    """
    if not res or not res.get("ids"):
        return []

    # Chroma’s “query” returns nested lists; “get” returns flat lists.
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

def _fetch_media_linked(chunk_ids: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Given a list of text‐chunk IDs, fetch all images/tables in Chroma whose
    `parent_chunk_id` is in that list. Returns two lists of dicts (imgs, tables).
    """
    if not chunk_ids:
        return [], []

    where_clause = {"parent_chunk_id": {"$in": chunk_ids}}
    imgs = _zip_ids_meta(collection_img.get(where=where_clause, include=["metadatas"]))
    tbls = _zip_ids_meta(collection_tbl.get(where=where_clause, include=["metadatas"]))
    return imgs, tbls

def _candidate_filters(meta: Dict) -> List[Dict]:
    """
    Given a parsed metadata‐dict from the LLM (e.g. {"Diseases":["Hepatitis"], "title":None, ...}),
    produce a list of “single‐field” where‐clauses to try in Chroma. E.g.:
      • if v is a list → use {"field": {"$in": v}}
      • if v is a non‐empty scalar → use {"field": {"$eq": v}}
    Finally always append a None (meaning “no metadata filter”).
    """
    out: List[Dict] = []
    for k, v in meta.items():
        if v in (None, "", [], {}):
            continue
        if isinstance(v, list):
            out.append({k: {"$in": v}})
        else:
            out.append({k: {"$eq": v}})
    out.append(None)  # fallback: no filter
    return out

from numpy import dot
from numpy.linalg import norm

def _top_media_by_similarity(question_vec: List[float],
                             media: Dict[str, Dict],
                             top_n: int = 2) -> List[str]:
    """
    Given a dict of media (key=media_id, value=metadata including “summary”),
    compute cosine similarity between question_vec and each media["summary"] embedding,
    return the top_n media_ids, sorted by descending similarity.
    """
    if not media:
        return []

    ids_list    = list(media.keys())
    summaries   = [media[mid]["summary"] for mid in ids_list]
    sum_vecs    = _embed(summaries)   # embeddings for all summaries
    q = question_vec
    sims = [ dot(q, v) / (norm(q)*norm(v) + 1e-9) for v in sum_vecs ]

    id_sims = list(zip(ids_list, sims))
    id_sims.sort(key=lambda t: t[1], reverse=True)
    return [mid for mid, _ in id_sims[:top_n]]

def smart_query(
        question: str,
        top_k: int = 3,
        return_media: bool = False   # ← new optional kw-arg
    ) -> str | tuple[str, list[tuple[str,str]]]: 
    """
    Perform a “smart” RAG:
     1) Metadata‐aware + semantic search in `scientific_chunks` to get top_k text chunks.
     2) Fetch media linked by chunk_id (images + tables).
     3) Semantic‐nearest search on `image_summaries` + `table_summaries` to add any “closest” media.
     4) Re‐rank all candidate media by cosine similarity of their summary embeddings (keep top1 image & top2 tables).
     5) Build a single Gemini prompt that contains:
         • The top text chunks (with chunk_id, title, authors, chunk preview).
         • A “## Linked images” section listing each figure’s 200-word summary, prefaced with `<<img:FULL_UUID>>`.
         • A “## Linked tables” section listing each table’s 200-word summary, prefaced with `<<tbl:FULL_UUID>>`.
     6) Send to Gemini. If Gemini needs to actually show a figure or table, it writes exactly `<<img:ID8>>` or `<<tbl:ID8>>`
        (8 hex chars) or the full UUID (36 chars). We catch either format, look up path, and render inline.
    """
    # ── 1) Embed question + attempt metadata filters one by one ───────────
    q_vec = _embed([question])[0]
    meta_raw = _safe_json(_gem_chat(_QUERY_PROMPT + question))
    hits_txt = None

    for flt in _candidate_filters(meta_raw):
        hits_txt = collection_txt.query(
            [q_vec],
            n_results=top_k,
            where=flt,
            include=["documents", "metadatas"]
        )
        # If we got at least one hit, keep that filter and break
        if hits_txt and hits_txt["ids"] and len(hits_txt["ids"][0]) > 0:
            break

    if not hits_txt or not hits_txt["ids"] or len(hits_txt["ids"][0]) == 0:
        # Fallback: pure semantic (no filter)
        hits_txt = collection_txt.query(
            [q_vec],
            n_results=top_k,
            where=None,
            include=["documents", "metadatas"]
        )

    docs  = hits_txt["documents"][0]
    metas = hits_txt["metadatas"][0]
    chunk_ids = [m["chunk_id"] for m in metas]

    # ── 2) Fetch media directly linked by chunk_id ───────────────────────
    imgs_link, tbls_link = _fetch_media_linked(chunk_ids)

    # ── 3) Semantic‐nearest search in media stores ──────────────────────
    imgs_sem_res = collection_img.query(
        [q_vec],
        n_results=top_k,
        include=["metadatas"]
    )
    tbls_sem_res = collection_tbl.query(
        [q_vec],
        n_results=top_k,
        include=["metadatas"]
    )
    imgs_sem = _zip_ids_meta(imgs_sem_res)
    tbls_sem = _zip_ids_meta(tbls_sem_res)

    # Combine linked + nearest, keyed by full “id”
    imgs_all = {m["id"]: m for m in (imgs_link + imgs_sem)}
    tbls_all = {t["id"]: t for t in (tbls_link + tbls_sem)}

    # ── 4) Re‐rank media by cosine similarity of their summary embeddings ──
    top_img_ids = _top_media_by_similarity(q_vec, imgs_all, 1)   # keep best 1 image
    top_tbl_ids = _top_media_by_similarity(q_vec, tbls_all, 2)   # keep best 2 tables

    imgs_final = {mid: imgs_all[mid] for mid in top_img_ids if mid in imgs_all}
    tbls_final = {tid: tbls_all[tid] for tid in top_tbl_ids if tid in tbls_all}

    # ── 5) Build Gemini prompt ─────────────────────────────────────────
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
            # Always show full 36-char UUID in the prompt
            ctx.append(f"* (img:{im['id']}) {im['summary']}")

    if tbls_final:
        ctx.append("\n## Linked tables")
        for tb in tbls_final.values():
            ctx.append(f"* (tbl:{tb['id']}) {tb['summary']}")

    full_prompt = textwrap.dedent(f"""
        You are given text chunks (academic paper extracts) plus
        concise summaries of images and tables that might belong to them.

        • Answer strictly using ONLY the provided material.
        • Cite chunks as (Doc 1), (Doc 2), etc.
        • If an image/table is essential, output exactly
            <<img:FULL_UUID>>   or   <<tbl:FULL_UUID>>
        on its own line (no other text).

        --- EXAMPLE 1 ---

        CONTEXT:
        ### Doc 1
        Title   : Hepatitis Subtype—Encoding
        ---
        “Each DNA sequence is converted via EIIP coding (A→0.1260, C→0.1340, G→0.0806, T→0.1335) to numeric form.”

        ### Doc 2
        Title   : Hepatitis Subtype—Transforms
        ---
        “After EIIP, a Discrete Sine Transform (DST) is applied, then a level-4 Haar wavelet, and SVD retains the top 5 singular vectors.”

        ## Linked images
        * (img:936a6f5a-b0d3-4526-9b7b-1c917e730a03) Pipeline flowchart showing EIIP→DST→Haar→SVD.

        ## Linked tables
        * (tbl:f062ae4e-c43a-4018-a400-7e9f3674f255) The tables displays the metrics of the models performance accross different combinations.

        QUESTION:
        Explain the signal-processing pipeline for hepatitis subtype classification.

    ANSWER:
    First, raw DNA is mapped via EIIP coding (Doc 1). Next, a Discrete Sine Transform (DST) is applied to those numeric vectors (Doc 2). Then a level 4 Haar wavelet extracts multiresolution coefficients (Doc 2). Finally, SVD is performed and the top 5 singular vectors feed into the classifier (Doc 2).  
    <<img:936a6f5a-b0d3-4526-9b7b-1c917e730a03>>
    In the above template you can see that the results part was excluded in the final answer as the question and the table id was not tagged as it was not specifically asking for that information.
                                                              
    > Follow the template accordigly without including any irrelevant information that might have been provided accidently and donot cite the table or image that is not relevant. 
    • Cite chunks as (Doc 1), (Doc 2)… .
    • If an image/table is essential, output exactly
        <<img:FULL_UUID>>   or   <<tbl:FULL_UUID>>
      on its own line (no other text on that line).

    --- MATERIAL ---
    {''.join(ctx)}
    --- END MATERIAL ---

    Question: "{question}"
    """)
    answer = _gem_chat(full_prompt)

    # ── 6) Inline render (Jupyter/VS Code) if Gemini emitted any media tokens ───
    #    We match either 8-hex chars OR full 36-char UUID (with hyphens).
    show: List[Tuple[str, str]] = []
    pattern = r"<<(img|tbl):([0-9A-Fa-f]{8}|[0-9A-Fa-f\-]{32,36})>>"
    for kind, token in re.findall(pattern, answer):
        kind = kind.lower()
        # If 8 hex chars, find the first media whose ID startswith token
        if len(token) == 8:
            if kind == "img":
                match = next((m for m in imgs_final.values() if m["id"].startswith(token)), None)
            else:
                match = next((t for t in tbls_final.values() if t["id"].startswith(token)), None)
        else:
            # 32–36 chars → treat as full UUID
            match = (imgs_final.get(token) if kind == "img" else tbls_final.get(token))

        if match:
            path = match["path"]
            if Path(path).exists():
                if show and (kind, path) in show:
                    pass
                else:
                    show.append((kind, path))

    # Display answer + inline media in Jupyter / VS Code if available
    try:
        from IPython.display import display, Markdown, Image
        display(Markdown(answer))
        for kind, p in show:
            if kind == "img":
                display(Image(filename=p))
            else:
                md_text = Path(p).read_text(encoding="utf-8")
                display(Markdown(md_text))
    except ImportError:
        # If not in notebook, just print text  paths
        print(answer)
        for kind, p in show:
            print(f"[{kind.upper()}]: {p}")

    # Return the tuple: (answer_text, list_of_(kind,path))
    if return_media:
        return answer, show
    else:
        answer