#Enterprise RAG - Streamlit UI
import os
import re
import html
import requests
import pandas as pd
import streamlit as st

# ⚙️Settings
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
LEVELS = ["public", "internal", "confidential"]
ROLE_LEVELS = {"employee" : ["public", "internal"], "manager" : LEVELS, "admin" : LEVELS}
ALL_SCOPE = "__all__"
EXAMPLES = [
    "What are the common failure modes of RAG?",
    "What is the difference between RAG and fine-tuning?",
    "How does metadata filtering protect private content?",
]
RESULT_LABELS = {
    "indexed" : "Indexed",
    "updated" : "Updated",
    "skipped" : "Already up to date",
    "empty" : "No readable text found",
}
MIME_TYPES = {
    ".pdf" : "application/pdf",
    ".docx" : "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt" : "text/plain",
    ".md" : "text/markdown",
}

st.set_page_config(page_title="Enterprise RAG", page_icon="🏢", layout="wide")

# 🎨Styling : answers read like footnoted prose, sources are numbered notes
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

html, body, .stApp, [data-testid="stSidebar"], [data-testid="stMarkdownContainer"] {font-family: 'IBM Plex Sans', system-ui, sans-serif;}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 980px;}

.page-title {font-size: 1.9rem; font-weight: 600; letter-spacing: -0.01em; line-height: 1.2; margin: 0 0 .25rem 0;}
.page-sub {opacity: .7; margin: 0 0 1.4rem 0; max-width: 62ch;}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] li {
    font-family: 'Source Serif 4', Georgia, serif; font-size: 1.06rem; line-height: 1.7;
}

.cite {display: inline-block; min-width: 1.2em; padding: 0 .32em; margin: 0 .12em; border-radius: 3px; text-align: center;
    background: var(--primary-color, #4C6EF5); color: #fff; font: 600 .68em/1.55 'IBM Plex Sans', sans-serif; vertical-align: super;}

.notes {margin-top: .9rem; padding-top: .7rem; border-top: 1px solid rgba(128,128,128,.3);}
.notes-title {font-size: .85rem; font-weight: 600; opacity: .75; margin-bottom: .6rem;}
.note {display: flex; gap: .7rem; align-items: flex-start; margin-bottom: .65rem;}
.note .cite {vertical-align: baseline; margin: .15rem 0 0 0; font-size: .78rem; flex-shrink: 0;}
.note-head {display: flex; flex-wrap: wrap; align-items: center; gap: .2rem .6rem; font-size: .92rem;}
.note-file {font-weight: 600;}
.note-page {opacity: .65; font-size: .82rem;}
.note details {margin-top: .15rem; font-size: .85rem;}
.note summary {cursor: pointer; opacity: .7;}
.note .quote {margin-top: .35rem; padding: .1rem 0 .1rem .8rem; border-left: 2px solid rgba(128,128,128,.45); opacity: .85; line-height: 1.55;
    font-family: 'Source Serif 4', Georgia, serif;}

.tag {display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: .74rem; font-weight: 600; margin: 0 6px 4px 0; border: 1px solid rgba(128,128,128,.4);}
.t-public {color: #2F9E6E; border-color: rgba(47,158,110,.55); background: rgba(47,158,110,.10);}
.t-internal {color: #4C6EF5; border-color: rgba(76,110,245,.55); background: rgba(76,110,245,.10);}
.t-confidential {color: #D9822B; border-color: rgba(217,130,43,.6); background: rgba(217,130,43,.10);}

.brand {font-weight: 600; font-size: 1.05rem; margin-bottom: .9rem;}
.who {font-weight: 600; font-size: 1rem; margin-bottom: .3rem;}
.small-label {font-size: .78rem; opacity: .65; margin: .7rem 0 .3rem 0;}
.dot {display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;}
.dot-on {background: #2F9E6E;}
.dot-off {background: #D64545;}

.steps {list-style: none; padding: 0; margin: 1.3rem 0 1.6rem 0; max-width: 62ch;}
.steps li {display: flex; gap: .85rem; margin-bottom: .9rem;}
.steps .cite {vertical-align: baseline; margin: .2rem 0 0 0; font-size: .8rem; flex-shrink: 0; height: 1.5em;}
.steps b {display: block;}
.steps .sub {opacity: .7; font-size: .93rem;}

[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.3); border-radius: 6px; padding: .8rem 1rem;}
</style>
""", unsafe_allow_html=True)

# 🔐Session state
defaults = {"api_key" : "", "user" : None, "messages" : [], "pending" : None, "login_error" : ""}
for name, value in defaults.items():
    if name not in st.session_state:
        st.session_state[name] = value

# 🧰Helpers
def call_api(method : str, path : str, **kwargs):
    headers = {"X-API-Key" : st.session_state.api_key}
    try:
        return requests.request(method, f"{API_URL}{path}", headers=headers, timeout=300, **kwargs)
    except requests.exceptions.RequestException:
        return None

@st.cache_data(ttl=15, show_spinner=False)
def fetch_documents(api_key : str) -> list:
    try:
        res = requests.get(f"{API_URL}/documents", headers={"X-API-Key" : api_key}, timeout=30)
        return res.json() if res.status_code == 200 else []
    except requests.exceptions.RequestException:
        return []

@st.cache_data(ttl=10, show_spinner=False)
def api_online() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False

def tag(text, kind : str = "plain") -> str:
    kind = re.sub(r"\W", "", str(kind))
    return f'<span class="tag t-{kind}">{html.escape(str(text))}</span>'

def error_detail(res) -> str:
    try:
        return str(res.json().get("detail", res.text))
    except ValueError:
        return res.text or f"Request failed with status {res.status_code}"

def scope_label(value : str) -> str:
    if value == ALL_SCOPE:
        return "Everything I can read"
    return "Company-wide documents" if value == "all" else value.title()

def page_header(title : str, sub : str):
    st.markdown(f'<p class="page-title">{html.escape(title)}</p><p class="page-sub">{html.escape(sub)}</p>', unsafe_allow_html=True)

# 💬Answer rendering : [1] markers in the text match the numbered notes below it
def format_answer(text : str) -> str:
    safe = html.escape(text, quote=False)
    def to_markers(match):
        numbers = [n.strip() for n in match.group(1).split(",")]
        return "".join(f'<span class="cite">{n}</span>' for n in numbers)
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", to_markers, safe)

def render_notes(sources : list):
    items = ""
    for s in sources:
        page = f'<span class="note-page">page {s["page"]}</span>' if s.get("page") else ""
        excerpt = html.escape(" ".join(s.get("snippet", "").split()))
        items += (
            '<div class="note">'
            f'<span class="cite">{s["id"]}</span>'
            '<div>'
            f'<div class="note-head"><span class="note-file">{html.escape(s["source"])}</span>{page}'
            f'{tag(s["department"], "dept")}{tag(s["access_level"], s["access_level"])}</div>'
            f'<details><summary>Show excerpt</summary><div class="quote">{excerpt}…</div></details>'
            '</div></div>'
        )
    st.markdown(f'<div class="notes"><div class="notes-title">Sources</div>{items}</div>', unsafe_allow_html=True)

def render_assistant(msg : dict):
    if msg.get("error"):
        st.error(msg["content"])
        return
    st.markdown(format_answer(msg["content"]), unsafe_allow_html=True)
    cited = re.search(r"\[\d", msg["content"]) is not None
    if cited and msg.get("sources"):
        render_notes(msg["sources"])
    elif not cited:
        st.caption("No passages were cited. The answer may not be in the documents you can read.")
    st.caption(f"Answered in {msg['latency'] / 1000:.1f} s")

def ask_backend(question : str, department) -> dict:
    res = call_api("POST", "/ask", json={"question" : question, "department" : department})
    if res is None:
        return {"role" : "assistant", "error" : True, "content" : f"Cannot reach the API at {API_URL}. Start it and ask again."}
    if res.status_code == 200:
        data = res.json()
        return {"role" : "assistant", "content" : data["answer"], "sources" : data["sources"], "latency" : data["latency_ms"]}
    if res.status_code == 401:
        return {"role" : "assistant", "error" : True, "content" : "Your session is no longer valid. Sign out and sign in again."}
    return {"role" : "assistant", "error" : True, "content" : error_detail(res)}

# 🔑Sign in and out
def sign_in():
    st.session_state.api_key = st.session_state.key_input.strip()
    st.session_state.login_error = ""
    res = call_api("GET", "/me")
    if res is None:
        st.session_state.login_error = f"Cannot reach the API at {API_URL}. Start it and try again."
    elif res.status_code == 200:
        st.session_state.user = res.json()
        st.session_state.messages = []
    elif res.status_code == 401:
        st.session_state.login_error = "That API key was not recognised. Check it and try again."
    elif res.status_code == 404:
        st.session_state.login_error = "The API has no /me endpoint yet. Add it to api.py and restart the API."
    else:
        st.session_state.login_error = f"The API returned an error ({res.status_code})."

def sign_out():
    st.session_state.api_key = ""
    st.session_state.user = None
    st.session_state.messages = []
    st.session_state.pending = None

def clear_chat():
    st.session_state.messages = []

def ask_example(question : str):
    st.session_state.pending = question

# 📚Sidebar
user = st.session_state.user
page = "Ask"
scope = ALL_SCOPE
with st.sidebar:
    st.markdown('<div class="brand">Enterprise RAG</div>', unsafe_allow_html=True)
    if user is None:
        st.text_input("API key", type="password", key="key_input", placeholder="Paste your API key", on_change=sign_in)
        st.button("Sign in", type="primary", on_click=sign_in)
        if st.session_state.login_error:
            st.error(st.session_state.login_error)
    else:
        levels = "".join(tag(level, level) for level in ROLE_LEVELS.get(user["role"], []))
        st.markdown(
            f'<div class="who">{html.escape(user["username"])}</div>'
            f'{tag(user["role"], "role")}{tag(user["department"], "dept")}'
            f'<div class="small-label">You can read</div>{levels}',
            unsafe_allow_html=True,
        )
        pages = ["Ask", "Documents"]
        if user["role"] in ("manager", "admin"):
            pages.append("Upload")
        if user["role"] == "admin":
            pages.append("Analytics")
        st.markdown('<div class="small-label">Go to</div>', unsafe_allow_html=True)
        page = st.radio("Go to", pages, label_visibility="collapsed")

        if page == "Ask":
            departments = sorted({d["department"] for d in fetch_documents(st.session_state.api_key)})
            st.markdown('<div class="small-label">Search in</div>', unsafe_allow_html=True)
            scope = st.selectbox("Search in", [ALL_SCOPE] + departments, format_func=scope_label, label_visibility="collapsed")
            if st.session_state.messages:
                transcript = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in st.session_state.messages)
                st.download_button("Export chat", transcript, file_name="enterprise-rag-chat.md")
                st.button("Clear chat", on_click=clear_chat)
        st.button("Sign out", on_click=sign_out)

    online = api_online()
    st.markdown(
        f'<div class="small-label"><span class="dot {"dot-on" if online else "dot-off"}"></span>'
        f'{"API online" if online else "API offline"}</div>',
        unsafe_allow_html=True,
    )
    st.caption(API_URL)

# 🚪Landing page (signed out)
if user is None:
    page_header("Ask your documents. See where each answer came from.",
                "Search company files in plain language. Every answer is numbered against the passages it used.")
    st.markdown(
        '<ol class="steps">'
        '<li><span class="cite">1</span><div><b>Search</b><span class="sub">Finds passages by meaning and by exact words, only in files your role can read.</span></div></li>'
        '<li><span class="cite">2</span><div><b>Rank</b><span class="sub">Puts the passages that best match your question first.</span></div></li>'
        '<li><span class="cite">3</span><div><b>Answer with sources</b><span class="sub">Writes the answer from those passages and numbers each source.</span></div></li>'
        '</ol>',
        unsafe_allow_html=True,
    )
    st.info("Sign in with your API key in the sidebar to start.")
    st.stop()

# 💬Ask
if page == "Ask":
    scope_text = "Searching everything you can read." if scope == ALL_SCOPE else f"Searching {scope_label(scope).lower()} only."
    page_header("Ask your documents", scope_text)

    question = st.chat_input("Ask a question about your documents")
    if st.session_state.pending:
        question = st.session_state.pending
        st.session_state.pending = None
    if question:
        st.session_state.messages.append({"role" : "user", "content" : question})

    if not st.session_state.messages:
        st.markdown('<div class="small-label">Start with a question</div>', unsafe_allow_html=True)
        for i, example in enumerate(EXAMPLES):
            st.button(example, key=f"example_{i}", on_click=ask_example, args=(example,))

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(msg)

    if question:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                reply = ask_backend(question, None if scope == ALL_SCOPE else scope)
            render_assistant(reply)
        st.session_state.messages.append(reply)

# 📚Documents
elif page == "Documents":
    page_header("Documents", "The files you are allowed to search.")
    docs = fetch_documents(st.session_state.api_key)
    if not docs:
        st.info("No documents are available to you yet.")
    else:
        df = pd.DataFrame(docs).rename(columns={"filename" : "File", "department" : "Department", "access_level" : "Access level", "chunks" : "Passages"})
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", len(df))
        c2.metric("Passages indexed", int(df["Passages"].sum()))
        c3.metric("Departments", df["Department"].nunique())

        f1, f2, f3 = st.columns([2, 1, 1])
        search = f1.text_input("File name contains", placeholder="policy")
        dept_filter = f2.multiselect("Department", sorted(df["Department"].unique()))
        level_filter = f3.multiselect("Access level", sorted(df["Access level"].unique()))
        view = df
        if search:
            view = view[view["File"].str.contains(search, case=False, na=False)]
        if dept_filter:
            view = view[view["Department"].isin(dept_filter)]
        if level_filter:
            view = view[view["Access level"].isin(level_filter)]
        st.dataframe(view[["File", "Department", "Access level", "Passages"]], hide_index=True)

# 📤Upload
elif page == "Upload":
    page_header("Upload documents", "Add files to the knowledge base. A file is re-indexed only when its content changes.")
    files = st.file_uploader("PDF, DOCX, TXT or MD files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    u1, u2 = st.columns(2)
    if user["role"] == "admin":
        department = u1.text_input("Department", value="all", help="Use all for company-wide documents.")
    else:
        department = u1.text_input("Department", value=user["department"], disabled=True, help="Managers upload to their own department.")
    level = u2.selectbox("Access level", LEVELS, index=1, help="Public and internal files are readable by every employee in the department. Confidential files are readable by managers and admins only.")

    if st.button("Upload and index", type="primary", disabled=not files):
        results = []
        bar = st.progress(0.0, text="Starting…")
        for i, f in enumerate(files):
            bar.progress(i / len(files), text=f"Indexing {f.name}…")
            ext = os.path.splitext(f.name)[1].lower()
            res = call_api(
                "POST", "/upload",
                files={"file" : (f.name, f.getvalue(), MIME_TYPES.get(ext, "application/octet-stream"))},
                data={"department" : department.strip().lower() or "all", "access_level" : level},
            )
            if res is None:
                outcome = f"Cannot reach the API at {API_URL}"
            elif res.status_code == 200:
                status = res.json().get("status", "")
                outcome = RESULT_LABELS.get(status, status)
            else:
                outcome = error_detail(res)
            results.append({"File" : f.name, "Result" : outcome})
        bar.progress(1.0, text="Finished")
        fetch_documents.clear()
        st.dataframe(pd.DataFrame(results), hide_index=True)

# 📊Analytics
elif page == "Analytics":
    page_header("Analytics", "How the assistant is being used.")
    res = call_api("GET", "/stats")
    if res is None:
        st.error(f"Cannot reach the API at {API_URL}.")
    elif res.status_code == 404:
        st.warning("The API has no /stats endpoint yet. Add it to api.py and restart the API.")
    elif res.status_code != 200:
        st.error(error_detail(res))
    else:
        stats = res.json()
        m1, m2, m3 = st.columns(3)
        m1.metric("Questions asked", stats["total_queries"])
        m2.metric("Average answer time", f"{stats['avg_latency_ms'] / 1000:.1f} s")
        m3.metric("People asking", len(stats["by_user"]))
        if stats["by_user"]:
            st.markdown('<div class="small-label">Questions per person</div>', unsafe_allow_html=True)
            st.bar_chart(pd.DataFrame(stats["by_user"]).rename(columns={"username" : "Person", "queries" : "Questions"}), x="Person", y="Questions", color="#4C6EF5")
        if stats["recent"]:
            st.markdown('<div class="small-label">Latest questions</div>', unsafe_allow_html=True)
            recent = pd.DataFrame(stats["recent"])
            recent["created_at"] = recent["created_at"].str[:16].str.replace("T", " ")
            recent["latency_ms"] = (recent["latency_ms"] / 1000).round(1)
            recent = recent.rename(columns={"created_at" : "Asked at", "username" : "Person", "question" : "Question", "latency_ms" : "Seconds"})
            st.dataframe(recent[["Asked at", "Person", "Question", "Seconds"]], hide_index=True)
