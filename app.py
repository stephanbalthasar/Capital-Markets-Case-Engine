# app.py
# Neon Case Tutor — Free LLM (Groq) + Web-Grounded Feedback & Chat
# - Free LLM via Groq (llama-3.1-8b/70b-instant): no credits or payments
# - Web retrieval from EUR-Lex, CURIA, ESMA, BaFin, Gesetze-im-Internet
# - Hidden model answer is authoritative; citations [1], [2] map to sources

import os
import re
import json
import hashlib
import pathlib
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
from urllib.parse import quote_plus, urlparse
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import requests
from bs4 import BeautifulSoup

# ---------------- Build fingerprint (to verify latest deployment) ----------------
APP_HASH = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:10]

# ---------------- Embeddings ----------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    """
    Try a small sentence-transformer; if unavailable (e.g., install timeouts), fall back to TF-IDF.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return ("sbert", model)
    except Exception:
        return ("tfidf", None)

def embed_texts(texts: List[str], backend):
    kind, model = backend
    if kind == "sbert":
        return model.encode(texts, normalize_embeddings=True)
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts)
    A = X.toarray()
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    return A / norms

def cos_sim(a, b):
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])

def split_into_chunks(text: str, max_words: int = 180):
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            chunks.append(" ".join(cur)); cur = []
    if cur: chunks.append(" ".join(cur))
    return chunks

# ---------------- Scoring Rubric ----------------
REQUIRED_ISSUES = [
    {
        "name": "Inside information & timing (Art 7(1),(2),(4) MAR); disclosure & delay (Art 17 MAR; Lafonta)",
        "points": 26,
        "keywords": ["art 7", "inside information", "precise nature", "intermediate step", "protracted process", "lafonta", "art 17", "as soon as possible", "delay", "mislead"],
    },
    {
        "name": "Prospectus requirement on admission (PR 2017/1129: Art 3(3); exemption Art 1(5)(a) ≤20%; approval Art 20; publication Art 21; MiFID II 4(1)(44))",
        "points": 18,
        "keywords": ["prospectus regulation", "art 3(3)", "admission to trading", "art 1(5)(a)", "20%", "article 20", "article 21", "mifid ii", "4(1)(44)"],
    },
    {
        "name": "Prospectus content & risk factors (PR Art 6(1) materiality; reasons 6(1)(c); risk factors Art 16(1))",
        "points": 12,
        "keywords": ["article 6(1)", "material information", "reasons for the issue", "article 16(1)", "risk factors"],
    },
    {
        "name": "Shareholding notifications (WpHG §§ 33, 34(2); acting in concert; §43 statement of intent; §44 sanctions)",
        "points": 18,
        "keywords": ["§ 33 wphg", "§ 34 wphg", "acting in concert", "gemeinschaftliches handeln", "§ 43 wphg", "§ 44 wphg"],
    },
    {
        "name": "Takeover law (WpÜG §§ 29(2), 30(2) control; §35 mandatory offer/disclosure; §59 suspension of rights)",
        "points": 16,
        "keywords": ["§ 29 wpüg", "§ 30 wpüg", "§ 35 wpüg", "mandatory offer", "control 30%", "§ 59 wpüg"],
    },
    {
        "name": "Clarification: subscription doesn’t trigger §38/33(3) WpHG for Neon; only Unicorn",
        "points": 10,
        "keywords": ["§ 38 wphg", "§ 33(3) wphg", "subscription", "neon", "unicorn"],
    },
]
DEFAULT_WEIGHTS = {"similarity": 0.4, "coverage": 0.6}

# ---------------- Robust keyword & citation checks ----------------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def canonicalize(s: str, strip_paren_numbers: bool = False) -> str:
    s = s.lower()
    s = s.replace("art.", "art").replace("article", "art").replace("–", "-")
    s = s.replace("wpüg", "wpüg")
    s = re.sub(r"\s+", "", s)
    if strip_paren_numbers:
        s = re.sub(r"\(\d+[a-z]?\)", "", s)
    s = re.sub(r"[^a-z0-9§]", "", s)
    return s

def keyword_present(answer: str, kw: str) -> bool:
    ans_can = canonicalize(answer, strip_paren_numbers=True)
    kw_can = canonicalize(kw, strip_paren_numbers=True)
    if kw.strip().lower().startswith(("§", "art")):
        return kw_can in ans_can
    return normalize_ws(kw).lower() in normalize_ws(answer).lower()

def coverage_score(answer: str, issue: Dict) -> Tuple[int, List[str]]:
    hits = [kw for kw in issue["keywords"] if keyword_present(answer, kw)]
    score = int(round(issue["points"] * (len(hits) / max(1, len(issue["keywords"])))))
    return score, hits

def detect_citation_issues(answer: str) -> Dict[str, List[str]]:
    issues, suggestions = [], []
    a = answer
    # Art 3(1) PR (public offer) vs Art 3(3) PR (admission to trading)
    if re.search(r"\bart\.?\s*3\s*\(\s*1\s*\)\s*(pr|prospectus)", a, flags=re.IGNORECASE):
        issues.append("You cited Art 3(1) PR for admission to trading (public-offer rule).")
        suggestions.append("For admission to a regulated market, cite Art 3(3) PR; also Art 20/21 PR on approval/publication.")
    # § 40 WpHG vs § 43(1) WpHG (statement of intent)
    if re.search(r"§\s*40\s*wphg", a, flags=re.IGNORECASE):
        issues.append("You cited § 40 WpHG. The statement of intent is § 43(1) WpHG.")
        suggestions.append("Replace § 40 WpHG with § 43(1) WpHG.")
    return {"issues": issues, "suggestions": suggestions}

def detect_substantive_flags(answer: str) -> List[str]:
    flags = []
    low = answer.lower()
    if "always delay" in low or re.search(r"\b(can|may)\s+always\s+delay\b", low):
        flags.append("Delay under Art 17(4) MAR is conditional: (a) legitimate interest, (b) not misleading, (c) confidentiality ensured.")
    return flags

def summarize_rubric(student_answer: str, model_answer: str, backend, required_issues: List[Dict], weights: Dict):
    embs = embed_texts([student_answer, model_answer], backend)
    sim = cos_sim(embs[0], embs[1])
    sim_pct = max(0.0, min(100.0, 100.0 * (sim + 1) / 2))

    per_issue, tot, got = [], 0, 0
    for issue in required_issues:
        pts = issue.get("points", 10)
        tot += pts
        sc, hits = coverage_score(student_answer, issue)
        got += sc
        per_issue.append({
            "issue": issue["name"], "max_points": pts, "score": sc,
            "keywords_hit": hits, "keywords_total": issue["keywords"],
        })
    cov_pct = 100.0 * got / max(1, tot)
    final = (weights["similarity"] * sim_pct + weights["coverage"] * cov_pct) / (weights["similarity"] + weights["coverage"])

    missing = []
    for row in per_issue:
        missed = [kw for kw in row["keywords_total"] if kw not in row["keywords_hit"]]
        if missed:
            missing.append({"issue": row["issue"], "missed_keywords": missed})

    citation_issues = detect_citation_issues(student_answer)
    substantive_flags = detect_substantive_flags(student_answer)

    return {
        "similarity_pct": round(sim_pct, 1),
        "coverage_pct": round(cov_pct, 1),
        "final_score": round(final, 1),
        "per_issue": per_issue,
        "missing": missing,
        "citation_issues": citation_issues,
        "substantive_flags": substantive_flags,
    }

# ---------------- Web Retrieval (RAG) ----------------
ALLOWED_DOMAINS = {
    "eur-lex.europa.eu",        # EU law (MAR, PR, MiFID II, TD)
    "curia.europa.eu",          # CJEU (Lafonta C‑628/13 etc.)
    "www.esma.europa.eu",       # ESMA guidelines/news
    "www.bafin.de",             # BaFin
    "www.gesetze-im-internet.de", "gesetze-im-internet.de",  # WpHG, WpÜG
    "www.bundesgerichtshof.de", # BGH
}

SEED_URLS = [
    "https://eur-lex.europa.eu/eli/reg/2014/596/oj",   # MAR
    "https://eur-lex.europa.eu/eli/reg/2017/1129/oj",  # Prospectus Regulation
    "https://eur-lex.europa.eu/eli/dir/2014/65/oj",    # MiFID II
    "https://eur-lex.europa.eu/eli/dir/2004/109/oj",   # Transparency Directive
    "https://curia.europa.eu/juris/liste.jsf?num=C-628/13",  # Lafonta
    "https://www.gesetze-im-internet.de/wphg/",
    "https://www.gesetze-im-internet.de/wpu_g/",
    "https://www.esma.europa.eu/press-news/esma-news/esma-finalises-guidelines-delayed-disclosure-inside-information-under-mar",
]

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

@st.cache_data(ttl=3600, show_spinner=False)
def duckduckgo_search(query: str, max_results: int = 6) -> List[Dict]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        r.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    out = []
    for a in soup.select("a.result__a"):
        href = a.get("href")
        title = a.get_text(" ", strip=True)
        if not href:
            continue
        domain = urlparse(href).netloc.lower()
        if any(domain.endswith(d) for d in ALLOWED_DOMAINS):
            out.append({"title": title, "url": href})
        if len(out) >= max_results:
            break
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_url(url: str) -> Dict:
    try:
        r = requests.get(url, headers=UA, timeout=25)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        title = (soup.title.get_text(strip=True) if soup.title else url)
        text = " ".join(soup.stripped_strings)
        text = re.sub(r"\s+", " ", text)
        if len(text) > 120000:
            text = text[:120000]
        return {"url": url, "title": title, "text": text}
    except Exception:
        return {"url": url, "title": url, "text": ""}

def build_queries(student_answer: str, extra_user_q: str = "") -> List[str]:
    base = [
        "Article 17 MAR delay disclosure ESMA guidelines site:eur-lex.europa.eu OR site:esma.europa.eu OR site:bafin.de",
        "Article 7(2) MAR precise information intermediate step Lafonta site:curia.europa.eu OR site:eur-lex.europa.eu",
        "Prospectus Regulation 2017/1129 Article 3(3) admission prospectus requirement site:eur-lex.europa.eu",
        "Prospectus Regulation Article 1(5)(a) 20% exemption site:eur-lex.europa.eu",
        "Prospectus Regulation Article 6(1) information Article 16(1) risk factors site:eur-lex.europa.eu",
        "MiFID II Article 4(1)(44) transferable securities site:eur-lex.europa.eu",
        "WpHG § 33 § 34(2) acting in concert gemeinschaftliches Handeln site:gesetze-im-internet.de OR site:bafin.de",
        "WpHG § 43 Abs 1 Absichtserklärung site:gesetze-im-internet.de OR site:bafin.de",
        "WpHG § 44 Rechte ruhen Sanktion site:gesetze-im-internet.de OR site:bafin.de",
        "WpÜG § 29 § 30 Kontrolle 30 Prozent acting in concert site:gesetze-im-internet.de OR site:bafin.de",
        "WpÜG § 35 Pflichtangebot Veröffentlichung BaFin site:gesetze-im-internet.de OR site:bafin.de",
        "WpÜG § 59 Ruhen von Rechten site:gesetze-im-internet.de OR site:bafin.de",
    ]
    if student_answer:
        base.append(f"({student_answer[:300]}) Neon Unicorn CFA MAR PR WpHG WpÜG site:eur-lex.europa.eu OR site:gesetze-im-internet.de")
    if extra_user_q:
        base.append(extra_user_q + " site:eur-lex.europa.eu OR site:gesetze-im-internet.de OR site:curia.europa.eu OR site:esma.europa.eu OR site:bafin.de")
    return base

def collect_corpus(student_answer: str, extra_user_q: str, max_fetch: int = 20) -> List[Dict]:
    results = [{"title": "", "url": u} for u in SEED_URLS]
    for q in build_queries(student_answer, extra_user_q):
        results.extend(duckduckgo_search(q, max_results=5))
    seen, cleaned = set(), []
    for r in results:
        url = r["url"]
        if url in seen:
            continue
        seen.add(url)
        domain = urlparse(url).netloc.lower()
        if not any(domain.endswith(d) for d in ALLOWED_DOMAINS):
            continue
        cleaned.append(r)
    fetched = []
    for r in cleaned[:max_fetch]:
        pg = fetch_url(r["url"])
        if pg["text"]:
            pg["title"] = pg["title"] or r.get("title") or r["url"]
            fetched.append(pg)
    return fetched

def retrieve_snippets(student_answer: str, model_answer: str, pages: List[Dict], backend, top_k_pages: int = 8, chunk_words: int = 170):
    import fitz  # PyMuPDF

def retrieve_snippets_with_manual(student_answer: str, model_answer: str, pages: List[Dict], backend, top_k_pages: int = 8, chunk_words: int = 170):
    # Load course manual
    try:
        doc = fitz.open("assets/EUCapML - Course Booklet.pdf")
        manual_text = " ".join([page.get_text() for page in doc])
        doc.close()
    except Exception as e:
        manual_text = ""
        st.warning(f"Could not load course manual: {e}")

    manual_chunks = split_into_chunks(manual_text, max_words=chunk_words)
    manual_meta = [(-1, "Course Manual", "EUCapML - Course Booklet.pdf")] * len(manual_chunks)

    # Prepare web chunks
    web_chunks, web_meta = [], []
    for i, p in enumerate(pages):
        for ch in split_into_chunks(p["text"], max_words=chunk_words):
            web_chunks.append(ch)
            web_meta.append((i, p["url"], p["title"]))

    all_chunks = manual_chunks + web_chunks
    all_meta = manual_meta + web_meta
    query = (student_answer or "") + "\n\n" + (model_answer or "")
    embs = embed_texts([query] + all_chunks, backend)
    qv, cvs = embs[0], embs[1:]
    sims = [cos_sim(qv, v) for v in cvs]
    idx = np.argsort(sims)[::-1]

    per_page = {}
    for j in idx[:400]:
        pi, url, title = all_meta[j]
        snip = all_chunks[j]
        arr = per_page.setdefault(pi, {"url": url, "title": title, "snippets": []})
        if len(arr["snippets"]) < 3:
            arr["snippets"].append(snip)
        if len(per_page) >= top_k_pages:
            break

    top_pages = [per_page[k] for k in sorted(per_page.keys())][:top_k_pages]
    source_lines = [f"[{i+1}] {tp['title']} — {tp['url']}" for i, tp in enumerate(top_pages)]
    return top_pages, source_lines

# ---------------- LLM via Groq (free) ----------------
def call_groq(messages: List[Dict], api_key: str, model_name: str = "llama-3.1-8b-instant",
              temperature: float = 0.2, max_tokens: int = 700) -> str:
    """
    Groq OpenAI-compatible chat endpoint. Models like llama-3.1-8b-instant / 70b-instant are free.
    """
    if not api_key:
        st.error("No GROQ_API_KEY found (add it to Streamlit Secrets).")
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(url, headers=headers, json=data, timeout=60)
        if r.status_code != 200:
            # Show exact error so it's easy to fix
            try: body = r.json()
            except Exception: body = r.text
            st.error(f"Groq error {r.status_code}: {body}")
            return None
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        st.error("Groq request timed out (60s). Try again or reduce max_tokens.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Groq request failed: {e}")
        return None

def system_guardrails():
    return (
        "You are a careful EU/German capital markets law tutor. "
        "Always ground your answers in the MODEL ANSWER (authoritative) AND the provided web SOURCES. "
        "If there is any conflict or doubt, follow the MODEL ANSWER and explain briefly. "
        "Cite sources as [1], [2], etc., matching the SOURCES list exactly. Be concise and didactic."
    )

def build_feedback_prompt(student_answer: str, rubric: Dict, model_answer: str, sources_block: str, excerpts_block: str) -> str:
    return f"""GRADE THE STUDENT'S ANSWER USING THE RUBRIC, DETECTED ISSUES, AND WEB SOURCES.

STUDENT ANSWER:
\"\"\"{student_answer}\"\"\"

RUBRIC SCORES:
- Similarity to model answer: {rubric['similarity_pct']}%
- Issue coverage: {rubric['coverage_pct']}%
- Overall score: {rubric['final_score']}%

DETECTED MIS-CITATIONS:
{json.dumps(rubric['citation_issues'], ensure_ascii=False)}

DETECTED SUBSTANTIVE FLAGS:
{json.dumps(rubric['substantive_flags'], ensure_ascii=False)}

MODEL ANSWER (AUTHORITATIVE):
\"\"\"{model_answer}\"\"\"

SOURCES (numbered):
{sources_block}

EXCERPTS (quote sparingly; use [n] to cite):
{excerpts_block}

TASK:
Provide <220 words of numbered, actionable feedback. Correct mis-citations (e.g., Art 3(1) PR -> Art 3(3) PR; § 40 WpHG -> § 43(1) WpHG).
Explain briefly why, with citations [n]. If sources diverge, follow the MODEL ANSWER.
"""

def build_chat_messages(chat_history: List[Dict], model_answer: str, sources_block: str, excerpts_block: str) -> List[Dict]:
    msgs = [{"role": "system", "content": system_guardrails()}]
    for m in chat_history[-8:]:
        if m["role"] in ("user", "assistant"): msgs.append(m)
    # Pin authoritative context and sources
    msgs.append({"role": "system", "content": "MODEL ANSWER (authoritative):\n" + model_answer})
    msgs.append({"role": "system", "content": "SOURCES:\n" + sources_block})
    msgs.append({"role": "system", "content": "RELEVANT EXCERPTS (quote sparingly):\n" + excerpts_block})
    return msgs

# ---------------- UI ----------------
import streamlit as st
import os
import requests

st.set_page_config(page_title="EUCapML Case Tutor", page_icon="⚖️", layout="wide")

def require_login():
    """Render only the PIN prompt until the user is authenticated."""
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("just_logged_in", False)

    if st.session_state.authenticated:
        return  # Already logged in

    # --- Login UI (only thing visible pre-auth) ---
    logo_col, title_col = st.columns([1, 5])
    with logo_col:
        logo_path = "assets/logo.png"
        try:
            if os.path.exists(logo_path):
                st.image(logo_path, width=240)
            else:
                st.markdown("### ⚖️ EUCapML Case Tutor")
        except Exception as e:
            st.markdown("### ⚖️ EUCapML Case Tutor")
            st.warning(f"Logo image could not be loaded: {e}")

    with title_col:
        st.title("EUCapML Case Tutor")

    pin_input = st.text_input("Enter your student PIN", type="password")

    try:
        correct_pin = st.secrets["STUDENT_PIN"]
    except KeyError:
        st.error("STUDENT_PIN not found in secrets. Configure it in .streamlit/secrets.toml.")
        st.stop()

    if pin_input and pin_input == correct_pin:
        st.session_state.authenticated = True
        st.session_state.just_logged_in = True
        st.success("PIN accepted. Loading…")
        st.rerun()

    # Not authenticated yet → show nothing else
    st.stop()

# Enforce login early — nothing else should render before this
require_login()

    if pin_input == correct_pin:
        st.session_state.authenticated = True
        st.session_state.just_logged_in = True
        st.success("PIN accepted. Click below to continue.")
        st.stop()  # Ends execution and triggers rerun automatically

# Show case selection only after login
if st.session_state.authenticated:
    def load_text_file(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"File '{filename}' not found."

    case_selection = st.selectbox("Choose a case", ["Case 1", "Case 2"])
    case_file_map = {
        "Case 1": ("case1.txt", "model_answer1.txt"),
        "Case 2": ("case2.txt", "model_answer2.txt")
    }
    case_filename, model_filename = case_file_map[case_selection]
    CASE = load_text_file(case_filename)
    MODEL_ANSWER = load_text_file(model_filename)

    if st.session_state.authenticated:
        with st.expander("📘 Case (click to read)"):
            st.write(CASE)

# Sidebar (visible to all users after login)
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("GROQ_API_KEY")
    if api_key:
        st.text_input("GROQ API Key", value="Provided via secrets/env", type="password", disabled=True)
    else:
        api_key = st.text_input("GROQ API Key", type="password", help="Set GROQ_API_KEY in Streamlit Secrets for production.")

    model_name = st.selectbox(
        "Model (free)",
        options=["llama-3.1-8b-instant", "llama-3.1-70b-instant"],
        index=0,
        help="Both are free; 8B is faster, 70B is smarter (and slower)."
    )
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.header("🌐 Web Retrieval")
    enable_web = st.checkbox("Enable web grounding", value=True)
    max_sources = st.slider("Max sources to cite", 3, 10, 6, 1)
    st.caption("DuckDuckGo HTML + filters to EUR‑Lex, CURIA, ESMA, BaFin, Gesetze‑im‑Internet, BGH.")

    st.divider()
    st.subheader("Diagnostics")
    if st.checkbox("Run Groq connectivity test"):
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key or ''}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": [{"role": "user", "content": "Say: hello from Groq test"}], "max_tokens": 8},
                timeout=20,
            )
            st.write("POST /chat/completions →", r.status_code)
            st.code((r.text or "")[:1000], language="json")
        except Exception as e:
            st.exception(e)

# Main UI
st.title("⚖️ EUCapML Case Tutor")
st.caption("Model answer prevails in doubt. Sources: EUR‑Lex, CURIA, ESMA, BaFin, Gesetze‑im‑Internet.")

if st.session_state.authenticated:
    with st.expander("📘 Case (click to read)"):
        st.write(CASE)

st.subheader("📝 Your Answer")
student_answer = st.text_area("Write your solution here (≥ ~120 words).", height=260)


# ------------- Actions -------------
colA, colB = st.columns([1, 1])

with colA:
    if st.button("🔎 Generate Feedback"):
        if len(student_answer.strip()) < 80:
            st.warning("Please write a bit more so I can evaluate meaningfully (≥ 80 words).")
        else:
            with st.spinner("Scoring and collecting sources..."):
                backend = load_embedder()
                rubric = summarize_rubric(student_answer, MODEL_ANSWER, backend, REQUIRED_ISSUES, DEFAULT_WEIGHTS)

                top_pages, source_lines = [], []
                if enable_web:
                    pages = collect_corpus(student_answer, "", max_fetch=22)
                    top_pages, source_lines = retrieve_snippets_with_manual(student_answer, MODEL_ANSWER, pages, backend, top_k_pages=max_sources, chunk_words=170)

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Semantic Similarity", f"{rubric['similarity_pct']}%")
            m2.metric("Issue Coverage", f"{rubric['coverage_pct']}%")
            m3.metric("Overall Score", f"{rubric['final_score']}%")

            # Breakdown
            with st.expander("🔬 Issue-by-issue breakdown"):
                for row in rubric["per_issue"]:
                    st.markdown(f"**{row['issue']}** — {row['score']} / {row['max_points']}")
                    st.markdown(f"- ✅ Found: {', '.join(row['keywords_hit']) if row['keywords_hit'] else '—'}")
                    miss = [kw for kw in row["keywords_total"] if kw not in row["keywords_hit"]]
                    st.markdown(f"- ⛔ Missing: {', '.join(miss) if miss else '—'}")

            # Deterministic corrections
            if rubric["citation_issues"]["issues"] or rubric["substantive_flags"]:
                st.markdown("### 🛠️ Detected corrections")
                for it in rubric["citation_issues"]["issues"]:
                    st.markdown(f"- ❗ {it}")
                for sg in rubric["citation_issues"]["suggestions"]:
                    st.markdown(f"  - ✔️ **Suggestion:** {sg}")
                for fl in rubric["substantive_flags"]:
                    st.markdown(f"- ⚖️ {fl}")

            # LLM narrative feedback
            sources_block = "\n".join(source_lines) if source_lines else "(no web sources available)"
            excerpts_items = []
            for i, tp in enumerate(top_pages):
                for sn in tp["snippets"]:
                    excerpts_items.append(f"[{i+1}] {sn}")
            excerpts_block = "\n\n".join(excerpts_items[: max_sources * 3]) if excerpts_items else "(no excerpts)"

            st.markdown("### 🧭 Narrative Feedback (with citations)")
            if api_key:
                messages = [
                    {"role": "system", "content": system_guardrails()},
                    {"role": "user", "content": build_feedback_prompt(student_answer, rubric, MODEL_ANSWER, sources_block, excerpts_block)},
                ]
                reply = call_groq(messages, api_key, model_name=model_name, temperature=temp, max_tokens=480)
                if reply:
                    st.write(reply)
                else:
                    st.info("LLM unavailable. See corrections above and the issue breakdown.")
            else:
                st.info("No GROQ_API_KEY found in secrets/env. Deterministic scoring and corrections shown above.")

            if source_lines:
                with st.expander("📚 Sources used"):
                    for line in source_lines:
                        st.markdown(f"- {line}")

with colB:
    st.markdown("### 💬 Tutor Chat (web‑grounded)")
    st.caption("Ask follow-up questions. Answers cite authoritative sources and follow the model answer.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if msg["role"] in ("user", "assistant"):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    user_q = st.chat_input("Ask a question about your feedback, the law, or how to improve…")
    if user_q:
        with st.spinner("Retrieving sources and drafting a grounded reply..."):
            backend = load_embedder()
            top_pages, source_lines = [], []
            if enable_web:
                pages = collect_corpus(student_answer, user_q, max_fetch=20)
                top_pages, source_lines = retrieve_snippets(
                    (student_answer or "") + "\n\n" + user_q,
                    MODEL_ANSWER, pages, backend, top_k_pages=max_sources, chunk_words=170
                )

            sources_block = "\n".join(source_lines) if source_lines else "(no web sources available)"
            excerpts_items = []
            for i, tp in enumerate(top_pages):
                for sn in tp["snippets"]:
                    excerpts_items.append(f"[{i+1}] {sn}")
            excerpts_block = "\n\n".join(excerpts_items[: max_sources * 3]) if excerpts_items else "(no excerpts)"

            st.session_state.chat_history.append({"role": "user", "content": user_q})

            if api_key:
                msgs = build_chat_messages(st.session_state.chat_history, MODEL_ANSWER, sources_block, excerpts_block)
                reply = call_groq(msgs, api_key, model_name=model_name, temperature=temp, max_tokens=600)
            else:
                reply = None

            if not reply:
                reply = (
                    "I couldn’t reach the LLM. Here are the most relevant source snippets:\n\n"
                    + (excerpts_block if excerpts_block != "(no excerpts)" else "— no sources available —")
                    + "\n\nIn doubt, follow the model answer."
                )

            with st.chat_message("assistant"):
                st.write(reply)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

st.divider()
st.markdown(
    "ℹ️ **Notes**: The app grounds answers in authoritative sources and the hidden model answer. "
    "If web sources appear to diverge, the tutor explains the divergence but follows the model answer."
)
