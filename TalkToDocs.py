import os
import re
import math
import streamlit as st

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

from config import PERSIST_DIR, EMBED_MODEL, LLM_MODEL, COLLECTION_NAME

# -------------------- UI --------------------
st.set_page_config(page_title="Talk to Your Documents", layout="wide")
st.title("📄 Talk to Your Documents")
os.makedirs(PERSIST_DIR, exist_ok=True)

# -------------------- Models & Vector Store --------------------
TOP_K                = 15
TEMPERATURE          = 0.2
USE_MMR              = False
MIN_STRONG_SCORE     = 0.70   # distance; lower is better
PAGE_WINDOW_DEFAULT  = 2
PAGE_WINDOW_EXPANDED = 4
MAX_PAGES_PER_SOURCE = 2
FOLLOWUP_SIM_THRESH  = 0.65   # cosine similarity for same-topic
QUESTION_CUES = ("?", "how", "what", "which", "who", "why", "where", "when", "by what", "how many", "how much", "rate", "percentage")

llm = OllamaLLM(model=LLM_MODEL, temperature=TEMPERATURE)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

# -------------------- Session --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "working_set" not in st.session_state:
    st.session_state.working_set = {
        "topic": "",
        "facets": [],
        "doc_ids": [],
        "sources": {},          # doc_id -> source path
        "page_windows": {},     # doc_id -> [min_page, max_page]
        "last_answer": "",
        "last_cited_pages": {}, # doc_id -> set(str(page))
    }

# -------------------- HISTORY detection --------------------
HISTORY_PATTERNS = [
    # first/last
    (re.compile(r"(how (did|do) (we|this) (start|begin))|(first (thing|message) i said)", re.I), "FIRST_USER"),
    (re.compile(r"what (was|is) the first (thing|message) (i|that i) said", re.I), "FIRST_USER"),
    (re.compile(r"(what (was|is) )?(my|the) last (question|message)\??", re.I), "LAST_USER"),
    (re.compile(r"what did i (ask|say) (last|previously)\??", re.I), "LAST_USER"),
    (re.compile(r"(what (was|is) )?(your|the) last (answer|reply)\??", re.I), "LAST_ASSISTANT"),
    (re.compile(r"what did you (say|answer) (last|previously)\??", re.I), "LAST_ASSISTANT"),
    # repeat/again variants
    (re.compile(r"\b(can\s*you|could\s*you|please)?\s*(repeat|say that again|say again)\b", re.I), "REPEAT_ASSISTANT"),
    (re.compile(r"\b(what did you (just )?say|come again|pardon|one more time)\b", re.I), "REPEAT_ASSISTANT"),
]

def detect_history_intent(message: str):
    msg = (message or "").strip()
    for pat, label in HISTORY_PATTERNS:
        if pat.search(msg):
            return label
    return None

def _strip_refs(text: str) -> str:
    if not text:
        return text
    return re.sub(r"\n\*\*References:\*\*[\s\S]*$", "", text).strip()

def get_first_user_message():
    for m in st.session_state.messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def get_last_user_message(exclude_current=True):
    msgs = st.session_state.messages[:-1] if exclude_current and st.session_state.messages else st.session_state.messages
    for m in reversed(msgs):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def get_last_assistant_message(clean=True):
    for m in reversed(st.session_state.messages):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            return _strip_refs(content) if clean else content
    return ""

def build_history_pairs():
    msgs = st.session_state.get("messages", [])
    pairs, human = [], None
    for m in msgs:
        if m.get("role") == "user":
            human = m.get("content", "")
        elif m.get("role") == "assistant":
            if human is not None:
                pairs.append((human, m.get("content", "")))
                human = None
    return pairs[-10:]

# -------------------- RAG helpers --------------------
def aggregate_source_pages(source_documents):
    per_docid_pages = {}
    docid_to_source = {}
    for d in source_documents or []:
        md = getattr(d, "metadata", {}) or {}
        did = md.get("doc_id")
        src = md.get("source", "Unknown")
        page = md.get("page", "N/A")
        if did:
            per_docid_pages.setdefault(did, set()).add(str(page))
            docid_to_source[did] = src
    return per_docid_pages, docid_to_source

def _soft_int(x):
    try:
        return int(x)
    except Exception:
        return None

def condense_pages(pages, max_pages=MAX_PAGES_PER_SOURCE):
    nums = [p for p in pages if str(p).isdigit()]
    nums_sorted = sorted(nums, key=lambda x: int(x))
    if nums_sorted:
        return nums_sorted[:max_pages]
    return list(pages)[:max_pages]

def format_ref_lines(docid_pages, docid_to_source):
    if not docid_pages:
        return ["- No document references for this reply."]
    lines = []
    for did in sorted(docid_pages):
        src = docid_to_source.get(did, "Unknown")
        pages = condense_pages(set(docid_pages[did]), MAX_PAGES_PER_SOURCE)
        lines.append(f"- `{src}` (pages {', '.join(pages)})")
    return lines

def build_filter_from_working_set(expanded=False):
    ws = st.session_state.working_set
    doc_ids     = ws.get("doc_ids", []) or []
    page_windows= ws.get("page_windows", {}) or {}
    if not doc_ids:
        return None
    or_terms = []
    for did in doc_ids:
        w = page_windows.get(did)
        if w and len(w) == 2:
            pad = PAGE_WINDOW_EXPANDED if expanded else PAGE_WINDOW_DEFAULT
            try:
                lo = max(1, int(w[0]) - pad)
                hi = int(w[1]) + pad
                or_terms.append({"doc_id": did, "page": {"$gte": lo, "$lte": hi}})
            except Exception:
                or_terms.append({"doc_id": did})
        else:
            or_terms.append({"doc_id": did})
    return {"$or": or_terms} if or_terms else {"doc_id": {"$in": doc_ids}}

def min_distance(query: str, flt, k: int = TOP_K):
    try:
        results = vectorstore.similarity_search_with_score(query, k=k, filter=flt)
    except Exception:
        return None
    if not results:
        return None
    return min(s for _, s in results)

# --- Semantic similarity for follow-up gating ---
def _cosine(a, b):
    if not a or not b:
        return 0.0
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)

def embed_similarity(text_a: str, text_b: str) -> float:
    try:
        va = embeddings.embed_query(text_a or "")
        vb = embeddings.embed_query(text_b or "")
        return _cosine(va, vb)
    except Exception:
        return 0.0

# --- Coverage check: do retrieved chunks even talk about the query terms? ---
_STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","is","it","that","this","those","these",
    "with","by","as","at","from","about","between","over","during","was","were","are","be","been",
    "how","what","why","which","who","whom","when","where"
}
def key_terms(q: str, max_terms=5):
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", q.lower())
    toks = [t for t in toks if t not in _STOPWORDS and len(t) > 2]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            out.append(t); seen.add(t)
        if len(out) >= max_terms:
            break
    return out

def coverage_sufficient(query: str, flt, k: int = 5, min_hits: int = 1) -> bool:
    try:
        results = vectorstore.similarity_search_with_score(query, k=k, filter=flt)
    except Exception:
        return False
    if not results:
        return False
    terms = key_terms(query, max_terms=5)
    if not terms:
        return True
    hits = 0
    for doc, _ in results:
        text = (doc.page_content or "").lower()
        if any(t in text for t in terms):
            hits += 1
            if hits >= min_hits:
                return True
    return False

# -------------------- Gates / Rewriter --------------------
_gate_prompt = PromptTemplate(
    template=(
        "Decide if the user message needs knowledge retrieval from uploaded documents.\n"
        "Output exactly one token: RETRIEVE or CHAT.\n\n"
        "Message:\n{message}\n\nDecision:"
    ),
    input_variables=["message"],
)
_followup_prompt = PromptTemplate(
    template=(
        "You have a prior conversation topic.\n"
        "Topic: {topic}\n"
        "User message: {message}\n\n"
        "If the message is a follow-up on that topic (e.g., 'explain more', pronouns like 'this/that', "
        "or clearly elaborating the same subject), output FOLLOWUP. Otherwise output NEW.\n\nDecision:"
    ),
    input_variables=["topic", "message"],
)
_rewrite_prompt = PromptTemplate(
    template=(
        "Rewrite the user's message into a single, standalone query for document retrieval.\n"
        "Use the topic and last answer to make it precise.\n\n"
        "Topic: {topic}\n"
        "Facets already covered: {facets}\n"
        "Last answer (for context only): {last_answer}\n"
        "User message: {message}\n\n"
        "Return only the rewritten query:"
    ),
    input_variables=["topic", "facets", "last_answer", "message"],
)

# History-aware small talk (always reads transcript; no retrieval/refs)
_smalltalk_prompt = PromptTemplate(
    template=(
        "Be brief, neutral, and friendly (<= 15 words). No references. No retrieval.\n"
        "Use the conversation context to stay coherent.\n\n"
        "Conversation start (first user line): {first_user}\n"
        "Last user line (before this): {last_user}\n"
        "Your last reply: {last_assistant}\n\n"
        "User: {message}\nReply:"
    ),
    input_variables=["first_user", "last_user", "last_assistant", "message"],
)

# REACT (emotional commentary) — acknowledge naturally, no retrieval/refs
_react_prompt = PromptTemplate(
    template=(
        "Respond briefly and naturally (<= 20 words) to the user's emotional reaction.\n"
        "Acknowledge the sentiment and, if helpful, tie it to the last answer in plain language.\n"
        "No retrieval. No references.\n\n"
        "Your last reply: {last_assistant}\n"
        "User: {message}\nReply:"
    ),
    input_variables=["last_assistant", "message"],
)

_topic_prompt = PromptTemplate(
    template=(
        "Create a 6-10 word topic label for the user's last question and your answer.\n"
        "Be specific (e.g., 'Global road-traffic deaths, 2010–2021 (WHO)').\n\n"
        "Question: {question}\nAnswer: {answer}\n\nTopic:"
    ),
    input_variables=["question", "answer"],
)
_facets_prompt = PromptTemplate(
    template=(
        "List 2-4 short bullets of the main facets covered in the answer.\n"
        "Return bullets separated by ' | ' with no extra text.\n\n"
        "Answer:\n{answer}\n\nFacets:"
    ),
    input_variables=["answer"],
)

def _invoke_clean(llm_obj, text: str) -> str:
    out = llm_obj.invoke(text) or ""
    return str(out).strip()

def sanitize_output(text: str) -> str:
    if not text:
        return text
    # nuke meta-phrases the model sometimes sneaks in
    text = re.sub(r"\bAccording to the Background Material,?\s*", "", text, flags=re.I)
    text = re.sub(r"\bthe provided chat history\b.*?\b", "", text, flags=re.I)
    text = re.sub(r"\bAccording to the text,?\s*", "", text, flags=re.I)
    # collapse accidental double spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def should_retrieve(message: str) -> bool:
    # override: if the user is clearly asking a question -> retrieve
    m = (message or "").strip().lower()
    if any(cue in m for cue in QUESTION_CUES):
        return True
    decision = _invoke_clean(llm, _gate_prompt.format(message=message)).upper()
    return "RETRIEVE" in decision

# ----- REACT detection (emotional comments that are NOT info-seeking) -----
_REACT_POS = re.compile(
    r"\b(wow+|wo+ah+|whoa+|omg|yikes|damn|geez|holy|insane|crazy|wild|unbelievable|scary|grim|sad|brutal)\b"
    r"|that'?s\s+(a lot|huge|insane|crazy|wild|high)"
    r"|\bis that (good|bad|high|low|normal)\b"
    r"|\ba lot\s+right\??\b",
    re.I
)
_INFO_SEEK = re.compile(
    r"\b(more|details?|explain|expand|why|how|break\s*down|by\s*region|numbers?|data|elaborate|compare|trend|change|percentage|rate|increase|decrease)\b",
    re.I
)
def detect_react_intent(message: str) -> bool:
    m = (message or "").strip().lower()
    if len(m) <= 60 and _REACT_POS.search(m) and not _INFO_SEEK.search(m):
        return True
    # short “right?” comments without info terms
    if len(m) <= 50 and m.endswith("?") and "right" in m and not _INFO_SEEK.search(m):
        return True
    return False

# ----- FOLLOW-UP detection (tightened) -----
def is_followup(topic: str, message: str) -> bool:
    msg = (message or "").strip().lower()
    if not topic:
        return False
    # explicit follow-up cues
    if any(w in msg for w in ["more", "else", "explain", "details", "why", "how", "break down", "elaborate", "what else", "expand", "elaboration"]):
        return True
    # pronoun-style reference + shortish
    if any(w in msg.split() for w in ["this","that","it","they","these","those","numbers"]) and len(msg) <= 80:
        return True
    # semantic similarity to topic label
    sim = embed_similarity(topic, message)
    return sim >= FOLLOWUP_SIM_THRESH

def rewrite_query(message: str, topic: str, facets, last_answer: str) -> str:
    facets_str = " | ".join(facets or [])
    return _invoke_clean(
        llm,
        _rewrite_prompt.format(
            topic=topic or "",
            facets=facets_str,
            last_answer=(last_answer or "")[:2000],
            message=message,
        ),
    )

def smalltalk_reply_with_history(message: str) -> str:
    return _invoke_clean(
        llm,
        _smalltalk_prompt.format(
            first_user=get_first_user_message(),
            last_user=get_last_user_message(exclude_current=True),
            last_assistant=get_last_assistant_message(clean=True),
            message=message,
        ),
    )

def react_reply(message: str) -> str:
    return _invoke_clean(
        llm,
        _react_prompt.format(
            last_assistant=get_last_assistant_message(clean=True),
            message=message,
        ),
    )

# -------------------- RAG Prompt --------------------
prompt_template = """
You are an expert, concise Retrieval-Augmented assistant. Follow these rules exactly.

[High-Level Policy]
- Answer accurately and directly using the Background Material when the question is about the uploaded content.
- If the material does not contain the answer, say you don’t know (do not invent).
- You may add widely-known, non-controversial basics if they help, but keep them brief.

[Follow-ups & Chat History]
- You have prior turns. Resolve pronouns (“this/that/it/they”) to the most recent relevant answer unless the user says otherwise.
- If the user asks for “more/expand/what else”, continue from your last answer and add depth/examples/implications.

[Numerical Synthesis]
- If asked for change “over/during” a period, compute the net change across that period even if sources list yearly changes. Provide the total; optionally the breakdown.

[Evidence Handling]
- Prefer synthesis over copying; quote only short key phrases if necessary.
- Extract numbers, names, steps precisely from the material.
- If sources conflict, say so briefly and summarize each view.

[Style]
- Plain, direct language. No filler. No “as an AI”. Do not mention “context”, “documents”, “Background Material”, or “chat history”.
- Use short lists or tight paragraphs.

[When Material Seems Irrelevant]
- If retrieved passages look off-topic or too weak, say you don’t know. Optionally suggest what to ask for.

[Output Format]
- Start with the answer.
- Optionally add a short “Key points” list when it improves clarity.

Background Material:
{context}

Chat History (reference only; don’t summarize unless asked):
{chat_history}

User Question:
{question}

Now answer:
"""
qa_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"],
)

# -------------------- Caption --------------------
try:
    st.caption(f"Indexed chunks: {vectorstore._collection.count()}")
except Exception:
    st.caption("Indexed chunks: —")

# -------------------- Render history --------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------- Chat --------------------
if user_q := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ws = st.session_state.working_set

            # A) HISTORY (first)
            history_intent = detect_history_intent(user_q)
            if history_intent:
                if history_intent == "FIRST_USER":
                    first = get_first_user_message()
                    resp = f"You started with: {first!r}" if first else "I don't have an earlier message from you."
                elif history_intent == "LAST_USER":
                    lastu = get_last_user_message(exclude_current=True)
                    resp = f"Your last message was: {lastu!r}" if lastu else "I don't see a previous message from you."
                elif history_intent == "LAST_ASSISTANT":
                    lasta = get_last_assistant_message(clean=True)
                    resp = f"My last answer was: {lasta!r}" if lasta else "I haven't answered anything yet."
                elif history_intent == "REPEAT_ASSISTANT":
                    lasta = get_last_assistant_message(clean=True)
                    resp = lasta if lasta else "I haven't said anything yet."
                else:
                    resp = "I checked the chat history, but I couldn’t resolve that request."
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
                st.stop()

            # B) REACT (emotional comment) — BEFORE follow-up
            if detect_react_intent(user_q):
                reply = react_reply(user_q)
                reply = sanitize_output(reply)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.stop()

            # C) FOLLOW-UP or NEW?
            topic = ws.get("topic", "")
            followup = is_followup(topic, user_q)

            if not followup:
                # CHAT vs RETRIEVE for brand-new messages
                if not should_retrieve(user_q):
                    reply = smalltalk_reply_with_history(user_q)   # history-aware small talk
                    reply = sanitize_output(reply)
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()

            # D) Build query for retrieval
            if followup and topic:
                query_for_retrieval = rewrite_query(user_q, ws.get("topic",""), ws.get("facets",[]), ws.get("last_answer","")) or user_q
            else:
                query_for_retrieval = user_q

            # E) Local-first scope + sufficiency check
            chosen_filter = None
            candidates = []
            if followup and ws.get("doc_ids"):
                candidates = [
                    build_filter_from_working_set(expanded=False),
                    build_filter_from_working_set(expanded=True),
                    {"doc_id": {"$in": ws.get("doc_ids", [])}},
                    None,  # global
                ]
            else:
                candidates = [None]  # global directly

            for cand in candidates:
                strong = (min_distance(query_for_retrieval, cand, k=TOP_K) or 1.0) < MIN_STRONG_SCORE
                covered = coverage_sufficient(query_for_retrieval, cand, k=5, min_hits=1)
                if strong and covered:
                    chosen_filter = cand
                    break
            if chosen_filter is None:
                chosen_filter = None  # fall back to global

            # F) Retrieve + Answer
            if USE_MMR:
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": TOP_K, "fetch_k": max(TOP_K*4, 10), "filter": chosen_filter}
                )
            else:
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": TOP_K, "filter": chosen_filter}
                )

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt, "document_variable_name": "context"},
            )
            chat_history_pairs = build_history_pairs()
            result = chain.invoke({"question": query_for_retrieval, "chat_history": chat_history_pairs})

            response = sanitize_output(result.get("answer") or result.get("result") or "")
            sources  = result.get("source_documents", []) or []
            st.markdown(response)

            per_docid_pages, docid_to_source = aggregate_source_pages(sources)

            # Only show refs if new pages/sources appear since last answer
            show_refs = False
            new_pages_map = {}
            last_cited = ws.get("last_cited_pages", {}) or {}
            for did, pages in per_docid_pages.items():
                prev = set(last_cited.get(did, set()))
                new_pages = set(pages) - prev
                if new_pages:
                    show_refs = True
                new_pages_map[did] = set(pages)

            if per_docid_pages and show_refs:
                ref_lines = format_ref_lines(per_docid_pages, docid_to_source)
                with st.expander("References", expanded=True):
                    for line in ref_lines:
                        st.markdown(line)
                refs_block = ("\n\n**References:**\n" + "\n".join(ref_lines))
            else:
                refs_block = ""

            st.session_state.messages.append({
                "role": "assistant",
                "content": response + refs_block
            })

            # G) Update working set
            try:
                topic_label = _invoke_clean(llm, _topic_prompt.format(question=user_q, answer=response))
            except Exception:
                topic_label = ws.get("topic", "")
            try:
                facets_raw = _invoke_clean(llm, _facets_prompt.format(answer=response))
                facets = [f.strip() for f in facets_raw.split("|") if f.strip()]
            except Exception:
                facets = ws.get("facets", [])

            new_doc_ids = list(per_docid_pages.keys())
            new_sources = {**ws.get("sources", {})}
            new_windows = {**ws.get("page_windows", {})}
            for did, pages in per_docid_pages.items():
                src = docid_to_source.get(did, "Unknown")
                new_sources[did] = src
                nums = [_soft_int(p) for p in pages if _soft_int(p)]
                if nums:
                    lo, hi = min(nums), max(nums)
                    new_windows[did] = [lo, hi]
                else:
                    new_windows.setdefault(did, [1, 9999])

            st.session_state.working_set = {
                "topic": topic_label or ws.get("topic", ""),
                "facets": facets or ws.get("facets", []),
                "doc_ids": sorted(list(set(ws.get("doc_ids", []) + new_doc_ids))),
                "sources": new_sources,
                "page_windows": new_windows,
                "last_answer": response[:4000],
                "last_cited_pages": {**last_cited, **{did: set(pages) for did, pages in per_docid_pages.items()}},
            }
