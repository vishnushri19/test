# app/main.py — FINAL
# Multi-source retrieval: Chroma F5 Docs → F5 Module-Aware Web Fallback → General Web Search
# Modules covered: LTM, GTM/DNS, APM, ASM/WAF, AFM, SSLO, F5OS, Automation
# Response style: Perplexity-like — always cites sources, never hallucinates commands
#
# Run with:
#   .\.venv312\Scripts\python.exe -m uvicorn app.main:app --reload

import os
import re
import json
import httpx
import urllib.parse
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any, Generator
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import psycopg2
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
import time

# ✅ Manifest-driven ops-first retriever (Chroma)
from f5_retriever import F5Retriever

# ✅ Command planner templates
from tmsh_templates import choose_template, extract_params, missing_fields

# Optional file parsing
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

load_dotenv()

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Vittu AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

GROQ_MODEL_DEFAULT = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─────────────────────────────────────────────────────────────
# F5 Retriever (Chroma)
# ─────────────────────────────────────────────────────────────
_f5_retriever = F5Retriever()


def search_f5_docs(query: str, top_k: int = 8, module_hint: str = "ltm") -> Dict[str, Any]:
    result = _f5_retriever.retrieve(query, module_hint=module_hint, k_ops=top_k, k_concepts=top_k)

    citations: List[Dict[str, Any]] = []
    seen = set()

    for c in result.chunks[:top_k]:
        md = c.metadata or {}
        key = (md.get("url", ""), md.get("module", ""), md.get("pack", ""),
               md.get("topic", ""), md.get("intent", ""), md.get("authority", ""))
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "title":     md.get("title", ""),
            "url":       md.get("url", ""),
            "module":    md.get("module", ""),
            "pack":      md.get("pack", ""),
            "topic":     md.get("topic", ""),
            "intent":    md.get("intent", ""),
            "authority": md.get("authority", ""),
        })

    context = _f5_retriever.format_for_prompt(result, max_chars_per_chunk=900)

    return {
        "picked_source": "chroma",
        "route":         result.route,
        "context":       context,
        "citations":     citations,
    }


# ─────────────────────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────────────────────
def get_db():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        result = urllib.parse.urlparse(database_url)
        return psycopg2.connect(
            host=result.hostname,
            port=result.port or 5432,
            dbname=result.path[1:],
            user=result.username,
            password=result.password,
        )
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "vittu_brain"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            pinned BOOLEAN DEFAULT FALSE,
            archived BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding FLOAT8[] NOT NULL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("ALTER TABLE knowledge_base ADD COLUMN IF NOT EXISTS content_hash TEXT;")

    cur.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    cur.execute("DROP TRIGGER IF EXISTS trg_sessions_updated_at ON sessions;")
    cur.execute("""
        CREATE TRIGGER trg_sessions_updated_at
        BEFORE UPDATE ON sessions
        FOR EACH ROW
        EXECUTE FUNCTION set_updated_at();
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);")
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_kb_content_hash
        ON knowledge_base(content_hash)
        WHERE content_hash IS NOT NULL;
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database initialized successfully")


init_db()


# ─────────────────────────────────────────────────────────────
# pgvector check
# ─────────────────────────────────────────────────────────────
def pgvector_available() -> bool:
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_available_extensions WHERE name='vector';")
        ok = cur.fetchone() is not None
        cur.close()
        conn.close()
        return bool(ok)
    except Exception:
        return False


PGVECTOR_OK = pgvector_available()


# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────
def normalize_text_for_hash(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def compute_content_hash(text: str) -> str:
    norm = normalize_text_for_hash(text)
    return hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()


# ─────────────────────────────────────────────────────────────
# F5 Query Detection — ALL modules
# ─────────────────────────────────────────────────────────────
def is_f5_query(query: str) -> bool:
    q = (query or "").lower()
    triggers = [
        # LTM
        "f5", "big-ip", "bigip", "ltm", "vip", "pool", "virtual server",
        "node", "monitor", "irule", "irules", "icontrol", "tmsh",
        "profiles", "ssl", "cipher", "clientssl", "serverssl",
        "snat", "persistence", "oneconnect", "http profile",
        "rate limit", "ha", "device trust", "configsync",
        "traffic-group", "failover", "self ip", "vlan", "route domain",
        # GTM / DNS
        "gtm", "big-ip dns", "wide ip", "wideip", "gslb", "topology",
        "datacenter", "gtm pool", "gtm server", "dns listener",
        "prober", "iquery",
        # APM
        "apm", "access policy", "vpe", "per-session", "per-request",
        "access profile", "webtop", "portal access", "network access",
        "saml", "oauth", "kerberos", "ldap", "radius", "mfa",
        "session variable", "logon page", "access control list",
        # ASM / AWAF
        "asm", "awaf", "waf", "security policy", "violation", "signature",
        "response page", "blocking", "bot", "dos",
        "brute force", "web scraping", "ip intelligence",
        "attack signature", "false positive",
        # AFM
        "afm", "firewall", "network firewall", "dos profile",
        "rule list", "port list", "address list",
        # SSLO
        "sslo", "ssl orchestrator", "ssl-orchestrator",
        "service chain", "inspection zone",
        # F5OS
        "f5os", "f5 os", "rseries", "r2000", "r4000", "r5000", "r10000",
        "velos", "chassis partition", "tenant", "f5os-a", "f5os-c",
        "blade", "platform layer", "tenant image", "tenant deployment", "tenant state",
        # Automation
        "as3", "declarative onboarding", "telemetry streaming",
        "automation toolchain", "ansible f5", "terraform f5",
    ]
    return any(t in q for t in triggers)


# ─────────────────────────────────────────────────────────────
# Module Router — ALL modules
# ─────────────────────────────────────────────────────────────
def infer_module_hint(query: str) -> str:
    q = (query or "").lower()

    # F5OS — most specific, check first
    if any(k in q for k in [
        "f5os", "f5 os", "rseries", "r2000", "r4000", "r5000", "r10000",
        "velos", "chassis partition", "f5os-a", "f5os-c",
        "blade", "platform layer", "tenant image", "tenant deployment", "tenant state"
    ]):
        return "f5os"

    # tenant alone could be F5OS context — check alongside other clues
    if "tenant" in q and any(k in q for k in ["f5os", "velos", "rseries", "partition"]):
        return "f5os"

    # SSLO
    if any(k in q for k in [
        "sslo", "ssl orchestrator", "ssl-orchestrator",
        "service chain", "inspection zone"
    ]):
        return "sslo"

    # WAF / ASM / AWAF
    if any(k in q for k in [
        "asm", "awaf", "waf", "security policy", "violation", "signature",
        "response page", "blocking", "bot", "brute force", "web scraping",
        "ip intelligence", "attack signature", "false positive"
    ]):
        return "waf"

    # APM
    if any(k in q for k in [
        "apm", "access policy", "vpe", "per-session", "per-request",
        "access profile", "webtop", "portal access", "network access",
        "saml", "oauth", "kerberos", "ldap", "radius", "mfa",
        "session variable", "logon page"
    ]):
        return "apm"

    # AFM
    if any(k in q for k in [
        "afm", "network firewall", "dos profile",
        "rule list", "port list", "address list"
    ]):
        return "afm"

    # GTM / DNS
    if re.search(r"\bgtm\b", q) or any(k in q for k in [
        "big-ip dns", "wide ip", "wideip", "gslb", "topology",
        "datacenter", "gtm pool", "gtm server", "dns listener",
        "prober", "iquery"
    ]):
        return "dns"

    # Automation
    if any(k in q for k in [
        "as3", "declarative onboarding", "telemetry streaming",
        "automation toolchain", "ansible f5", "terraform f5"
    ]):
        return "automation"

    # Platform
    if any(k in q for k in ["license", "provision", "tm.platform"]):
        return "platform"

    # LTM — default
    return "ltm"


# ─────────────────────────────────────────────────────────────
# Authoritative Command Guardrails
# ─────────────────────────────────────────────────────────────
def is_command_intent(query: str) -> bool:
    q = (query or "").lower()
    triggers = [
        "tmsh", "icontrol", "/mgmt/", "rest api", "api endpoint",
        "create ", "modify ", "delete ", "add ", "remove ",
        "how do i configure", "how to configure",
        "give me the command", "exact command", "cli command",
        "run this", "show ", "list ",
    ]
    return any(t in q for t in triggers)


def is_mutating_command_request(query: str) -> bool:
    q = (query or "").lower()
    mutators = [
        "create ", "modify ", "delete ", "add ", "remove ",
        "set ", "replace ", "enable ", "disable ",
        "attach ", "assign ", "apply ",
    ]
    return any(m in q for m in mutators)


def has_authoritative_ops_evidence_for_module(
    citations: List[Dict[str, Any]], expected_module: str
) -> bool:
    em = (expected_module or "").strip().lower()
    for c in citations or []:
        if (c.get("pack") == "core_ops") and (c.get("authority") == "clouddocs"):
            cm = (c.get("module") or "").strip().lower()
            if not em or not cm:
                continue
            if cm == em:
                return True
    return False


def build_command_block_response(module_hint: str) -> str:
    return (
        f"I can't safely give an exact create/modify/delete tmsh command for this request "
        f"with the current knowledge packs loaded for module='{module_hint}', because I don't "
        f"have authoritative CLI syntax evidence for that module in context.\n\n"
        "What I can do right now:\n"
        "1) Ask the minimum details needed (names, destination/members, profiles, SNAT/persistence).\n"
        "2) Provide non-destructive validation commands (show/list) to inspect current state.\n\n"
        "Reply with:\n"
        "- BIG-IP version (major)\n"
        "- Object type (virtual/pool/node/profile/policy)\n"
        "- Names (virtual name, pool name)\n"
        "- Destination IP:port (for virtual)\n"
        "- Member IP:port list (for pool)\n"
        "- Profiles required (clientssl/serverssl/http/tcp)\n"
        "- SNAT requirement (automap or snatpool) and persistence type\n\n"
        "Safe validation examples:\n"
        "- tmsh list ltm virtual\n"
        "- tmsh list ltm pool\n"
        "- tmsh show ltm virtual all-properties\n"
        "- tmsh show ltm pool all-properties\n"
    )


def tmsh_output_looks_valid(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if "tmsh" not in t and not any(k in t for k in [
        "create ltm", "modify ltm", "delete ltm", "show ltm", "list ltm"
    ]):
        return False
    if "i'm not sure" in t or "guess" in t:
        return False
    return True


# ─────────────────────────────────────────────────────────────
# Session Helpers
# ─────────────────────────────────────────────────────────────
def ensure_session_exists(session_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (session_id, title) VALUES (%s, %s) ON CONFLICT (session_id) DO NOTHING",
        (session_id, session_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def touch_session(session_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
        (session_id,)
    )
    conn.commit()
    cur.close()
    conn.close()


def maybe_autotitle_session(session_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT title FROM sessions WHERE session_id = %s", (session_id,))
    row = cur.fetchone()
    if not row or (row[0] or "") != session_id:
        cur.close()
        conn.close()
        return
    cur.execute(
        "SELECT content FROM conversations WHERE session_id = %s AND role='user' ORDER BY timestamp ASC LIMIT 1",
        (session_id,),
    )
    m = cur.fetchone()
    if not m:
        cur.close()
        conn.close()
        return
    first = (m[0] or "").strip()
    title = " ".join(first.split()[:8])[:60].strip() or "New chat"
    cur.execute("UPDATE sessions SET title=%s WHERE session_id=%s", (title, session_id))
    conn.commit()
    cur.close()
    conn.close()


# ─────────────────────────────────────────────────────────────
# Conversation Store
# ─────────────────────────────────────────────────────────────
def save_message(session_id: str, role: str, content: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversations (session_id, role, content) VALUES (%s, %s, %s)",
        (session_id, role, content)
    )
    conn.commit()
    cur.close()
    conn.close()


def get_history(session_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content, timestamp FROM conversations WHERE session_id=%s ORDER BY timestamp ASC LIMIT %s",
        (session_id, limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"role": r, "content": c, "timestamp": ts.isoformat()} for r, c, ts in rows]


def get_history_for_llm(session_id: str, limit: int = 8) -> List[Dict[str, str]]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM conversations WHERE session_id=%s ORDER BY timestamp DESC LIMIT %s",
        (session_id, limit),
    )
    rows = list(reversed(cur.fetchall()))
    cur.close()
    conn.close()
    return [{"role": r, "content": c} for r, c in rows]


# ─────────────────────────────────────────────────────────────
# RAG (Postgres Memory KB)
# ─────────────────────────────────────────────────────────────
def should_use_rag(query: str) -> bool:
    q = query.lower().strip()
    personal_triggers = [
        "my name", "what is my name", "who am i", "my favorite",
        "what is my favorite", "remember this", "remember that",
        "about me", "my preference", "i like", "i love", "i prefer",
    ]
    return not any(t in q for t in personal_triggers)


def search_knowledge_base_python(
    query: str, top_k: int = 3, min_score: float = 0.35
) -> List[str]:
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT content, embedding FROM knowledge_base")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return []

        qemb = embedding_model.encode(query).tolist()
        qn = float(np.linalg.norm(qemb) + 1e-10)

        scored: list[tuple[float, str]] = []
        for content, emb in rows:
            en = float(np.linalg.norm(emb) + 1e-10)
            sim = float(np.dot(qemb, emb) / (qn * en))
            scored.append((sim, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[:top_k] if s >= float(min_score)]
    except Exception as e:
        print(f"RAG error: {e}")
        return []


def search_knowledge_base(
    query: str, top_k: int = 3, min_score: float = 0.35
) -> List[str]:
    return search_knowledge_base_python(query, top_k=top_k, min_score=min_score)


# ─────────────────────────────────────────────────────────────
# Web Search — General (news, CVEs, time-sensitive)
# ─────────────────────────────────────────────────────────────
def needs_web_search(query: str) -> bool:
    q = query.lower().strip()

    skip_keywords = [
        "what day", "what date", "what time", "what year",
        "day today", "date today", "time now", "day is it",
        "what is today", "current time",
    ]
    if any(k in q for k in skip_keywords):
        return False

    search_keywords = [
        "latest", "recent", "today", "this week", "this month",
        "right now", "currently", "just released", "2026", "2025",
        "news", "announcement", "update", "updates", "release",
        "released", "launched", "new version", "what happened",
        "cve", "vulnerability", "vulnerabilities", "patch", "patches",
        "exploit", "zero-day", "breach", "attack", "threat", "advisory",
        "price", "stock", "weather", "score", "result",
        "who is", "what is", "how does", "where is", "when did",
        "tell me about", "explain", "are you up to date",
        "up to date", "do you know about",
    ]
    return any(kw in q for kw in search_keywords)


def web_search(query: str) -> str:
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
        context_parts = []
        if response.get("answer"):
            context_parts.append(f"Direct Answer: {response['answer']}")
        for r in response.get("results", []):
            title = r.get("title", "Result")
            url = r.get("url", "")
            content = (r.get("content") or "")[:400]
            context_parts.append(f"[{title}] ({url}):\n{content}")
        result = "\n\n".join(context_parts)
        print(f"✅ General web search returned {len(response.get('results', []))} results")
        return result
    except Exception as e:
        print(f"Tavily error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# F5 Module-Aware Web Fallback
# Fires when Chroma context is thin — searches official F5 sources
# per module: LTM, GTM, APM, WAF, AFM, SSLO, F5OS, Automation
# ─────────────────────────────────────────────────────────────
F5_MODULE_SEARCH_SCOPE = {
    "ltm":        "site:my.f5.com OR site:community.f5.com LTM",
    "waf":        "site:my.f5.com OR site:community.f5.com ASM AWAF WAF",
    "apm":        "site:my.f5.com OR site:community.f5.com APM \"access policy\"",
    "afm":        "site:my.f5.com OR site:community.f5.com AFM \"network firewall\"",
    "dns":        "site:my.f5.com OR site:community.f5.com GTM \"BIG-IP DNS\" GSLB",
    "sslo":       "site:my.f5.com OR site:community.f5.com \"SSL Orchestrator\" SSLO",
    "f5os":       "site:my.f5.com OR site:community.f5.com \"F5OS\" OR \"rSeries\" OR \"VELOS\" tenant",
    "automation": "site:my.f5.com OR site:community.f5.com AS3 \"declarative onboarding\" automation",
    "platform":   "site:my.f5.com OR site:community.f5.com \"BIG-IP\" license provisioning",
}


def needs_f5_web_fallback(query: str, f5_context: str) -> bool:
    """Fire when query is F5-related but Chroma returned thin/empty context."""
    if not is_f5_query(query):
        return False
    return len((f5_context or "").strip()) < 300


def web_search_f5_targeted(query: str, module_hint: str = "ltm") -> str:
    """
    Targeted search scoped to official F5 sources for the specific module.
    Automatically extracts K-article numbers from URLs.
    """
    try:
        scope = F5_MODULE_SEARCH_SCOPE.get(module_hint, "site:my.f5.com OR site:community.f5.com")
        f5_query = f"{scope} {query}"

        response = tavily_client.search(
            query=f5_query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )

        context_parts = []
        if response.get("answer"):
            context_parts.append(f"Direct Answer: {response['answer']}")

        for r in response.get("results", []):
            title = r.get("title", "Result")
            url = r.get("url", "")
            content = (r.get("content") or "")[:500]
            # Auto-extract K-article number for citations
            k_article = ""
            k_match = re.search(r"K\d{5,}", url)
            if k_match:
                k_article = f" [{k_match.group()}]"
            context_parts.append(f"[{title}{k_article}] ({url}):\n{content}")

        result = "\n\n".join(context_parts)
        print(
            f"✅ F5 [{module_hint.upper()}] web fallback returned "
            f"{len(response.get('results', []))} results for: {query[:60]}"
        )
        return result
    except Exception as e:
        print(f"F5 targeted web search error [{module_hint}]: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# System Prompt — Perplexity-style: always shows sources used
# ─────────────────────────────────────────────────────────────
def build_system_prompt(
    f5_context: str,
    rag_context: str,
    web_context: str,
    planner_context: str,
    module_hint: str = "ltm",
    f5_web_fallback_used: bool = False,
) -> str:
    now = datetime.now()
    current_date = now.strftime("%A, %B %d, %Y")
    current_time = now.strftime("%I:%M %p")

    sources_used = []
    if f5_context:
        sources_used.append(f"📘 F5 Docs — Chroma [{module_hint.upper()}]")
    if rag_context:
        sources_used.append("📚 Local Memory KB")
    if web_context and f5_web_fallback_used:
        sources_used.append(f"🌐 F5 Web [{module_hint.upper()}] (my.f5.com / DevCentral)")
    elif web_context:
        sources_used.append("🌐 Live Web Search")
    if not sources_used:
        sources_used.append("💡 General LLM knowledge only — no verified F5 source found")

    source_line = " + ".join(sources_used)

    prompt = f"""You are Vittu, an expert F5 Networks AI assistant — modeled after Perplexity AI.

TODAY: {current_date} | TIME: {current_time}
MODULE: {module_hint.upper()}
SOURCES USED: {source_line}

═══════════════════════════════════════════════
RESPONSE RULES — FOLLOW EXACTLY:
═══════════════════════════════════════════════

A) Default behavior (non-troubleshooting):
- Use sources and cite them inline when you are stating specific facts, product behavior, or exact syntax.
- If no verified source is available, say so plainly.

B) Troubleshooting behavior (VIP down / timeout / reset / 503 / SSL handshake / intermittent):
- Do NOT start with sources. Start with failure isolation.
- Use this structure:
  1) Symptom classification (timeout vs reset vs 503 vs TLS fail)
  2) Layer isolation: Client → F5 ingress → F5 selection → F5 egress → Server → Return path
  3) Decision points: “If you see X, it means Y; do Z next”
  4) Likely root causes (ranked)
  5) Sources (at the end)

C) Command safety / evidence:
- You MAY provide non-destructive troubleshooting commands even if they are not present verbatim in context:
  - tmsh show/list, tail, tcpdump, openssl s_client
- You MUST NOT provide create/modify/delete commands, iRules, or exact GUI click paths unless the authoritative context below contains the exact syntax/path.

D) Always use numbered steps for procedures.
E) End every answer with:
   📎 Sources: [list the URLs/K-articles used] or "None found — recommend checking my.f5.com"


═══════════════════════════════════════════════

AUTHORITATIVE COMMAND RULES (NON-NEGOTIABLE):
═══════════════════════════════════════════════
- Only provide create/modify/delete commands when the F5 Docs Context below
  contains authoritative CLI syntax (clouddocs, core_ops) for the SAME MODULE.
- If evidence is missing → ask for missing details or give read-only show/list commands only.
- tmsh format when you do provide it:
  1) Evidence (cite 1-2 sources from context)
  2) Command (code block)
  3) Verification (code block)
  4) Notes / assumptions
"""

    if planner_context:
        prompt += f"\n\n{planner_context}"

    if f5_context:
        prompt += f"\n\n### F5 Docs Context [{module_hint.upper()}] — HIGHEST PRIORITY:\n{f5_context}"

    if rag_context:
        prompt += f"\n\n### Local Memory KB:\n{rag_context}"

    if web_context:
        label = f"F5 Web Results [{module_hint.upper()}] (my.f5.com / DevCentral)" \
            if f5_web_fallback_used else "Live Web Search Results"
        prompt += f"\n\n### {label}:\n{web_context}"
    else:
        prompt += "\n\n### Note: No web search performed for this query."

    return prompt


# ─────────────────────────────────────────────────────────────
# Local Ollama
# ─────────────────────────────────────────────────────────────
async def get_ollama_response(messages: list) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.post(
                "http://localhost:11434/api/chat",
                json={"model": "vittu", "messages": messages, "stream": False},
            )
            data = response.json()
            return data["message"]["content"]
    except Exception as e:
        print(f"Ollama error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# API Models
# ─────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    messages: List[Message]
    use_local: bool = False
    stream: bool = False


class RenameSessionRequest(BaseModel):
    session_id: str
    title: str


class SessionActionRequest(BaseModel):
    session_id: str


class KBQuery(BaseModel):
    query: str
    top_k: int = 8


# ─────────────────────────────────────────────────────────────
# KB Query Endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/kb/query")
def kb_query(payload: KBQuery):
    mod = infer_module_hint(payload.query)
    result = search_f5_docs(payload.query, top_k=payload.top_k, module_hint=mod)
    return {
        "picked_source":   result.get("picked_source"),
        "route":           result.get("route"),
        "module_hint":     mod,
        "citations":       result.get("citations", []),
        "context_preview": (result.get("context") or "")[:2500],
    }


# ─────────────────────────────────────────────────────────────
# Session Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/sessions")
async def list_sessions(include_archived: bool = False):
    conn = get_db()
    cur = conn.cursor()
    if include_archived:
        cur.execute("""
            SELECT session_id, COALESCE(title, session_id),
                   pinned, archived, updated_at
            FROM sessions
            ORDER BY pinned DESC, updated_at DESC LIMIT 300
        """)
    else:
        cur.execute("""
            SELECT session_id, COALESCE(title, session_id),
                   pinned, archived, updated_at
            FROM sessions
            WHERE archived = FALSE
            ORDER BY pinned DESC, updated_at DESC LIMIT 300
        """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"sessions": [
        {
            "session_id": sid,
            "title": title,
            "pinned": bool(pinned),
            "archived": bool(archived),
            "updated_at": updated_at.isoformat() if updated_at else None,
        }
        for sid, title, pinned, archived, updated_at in rows
    ]}


@app.post("/sessions/new")
async def new_session():
    sid = f"vittu-{int(datetime.now().timestamp() * 1000)}"
    ensure_session_exists(sid)
    touch_session(sid)
    return {"session_id": sid}


@app.post("/sessions/rename")
async def rename_session(req: RenameSessionRequest):
    title = (req.title or "").strip()[:80]
    if not title:
        return {"ok": False, "error": "title_required"}
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET title=%s WHERE session_id=%s", (title, req.session_id))
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True}


@app.post("/sessions/pin")
async def pin_session(req: SessionActionRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET pinned=TRUE WHERE session_id=%s", (req.session_id,))
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True}


@app.post("/sessions/unpin")
async def unpin_session(req: SessionActionRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET pinned=FALSE WHERE session_id=%s", (req.session_id,))
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True}


@app.post("/sessions/archive")
async def archive_session(req: SessionActionRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET archived=TRUE WHERE session_id=%s", (req.session_id,))
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True}


@app.post("/sessions/unarchive")
async def unarchive_session(req: SessionActionRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET archived=FALSE WHERE session_id=%s", (req.session_id,))
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True}


@app.post("/sessions/delete")
async def delete_session(req: SessionActionRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM conversations WHERE session_id=%s", (req.session_id,))
    cur.execute("DELETE FROM sessions WHERE session_id=%s", (req.session_id,))
    conn.commit()
    cur.close()
    conn.close()
    return {"ok": True}


@app.get("/history/{session_id}")
async def history(session_id: str):
    return {"session_id": session_id, "messages": get_history(session_id, limit=500)}


# ─────────────────────────────────────────────────────────────
# Knowledge Upload + Ingestion
# ─────────────────────────────────────────────────────────────
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i: i + max_chars].strip())
        i += max_chars - overlap
    return [c for c in chunks if len(c) > 40]


def extract_text_from_upload(file: UploadFile, raw: bytes) -> str:
    name = (file.filename or "").lower()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if not pypdf:
            raise HTTPException(status_code=400, detail="pypdf not installed. Run: pip install pypdf")
        reader = pypdf.PdfReader(BytesIO(raw))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    if name.endswith(".docx"):
        if not docx:
            raise HTTPException(status_code=400, detail="python-docx not installed. Run: pip install python-docx")
        d = docx.Document(BytesIO(raw))
        return "\n".join([p.text for p in d.paragraphs])
    raise HTTPException(status_code=400, detail="Unsupported file type. Use .txt, .pdf, or .docx")


@app.post("/knowledge")
async def add_knowledge(content: str, source: str = "manual"):
    content = (content or "").strip()
    if not content:
        return {"status": "skipped", "reason": "empty_content"}
    content_hash = compute_content_hash(content)
    embedding = embedding_model.encode(content).tolist()
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO knowledge_base (content, embedding, source, content_hash) VALUES (%s,%s,%s,%s) ON CONFLICT (content_hash) DO NOTHING",
        (content, embedding, source, content_hash),
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "added", "source": source}


@app.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...), source: str = Form("upload")):
    try:
        raw = await file.read()
        text = extract_text_from_upload(file, raw)
        chunks = chunk_text(text, max_chars=1200, overlap=150)
        if not chunks:
            return {"ok": False, "error": "no_text_found"}

        conn = get_db()
        cur = conn.cursor()
        inserted = 0
        for c in chunks[:200]:
            c = c.strip()
            if not c:
                continue
            h = compute_content_hash(c)
            emb = embedding_model.encode(c).tolist()
            cur.execute(
                "INSERT INTO knowledge_base (content, embedding, source, content_hash) VALUES (%s,%s,%s,%s) ON CONFLICT (content_hash) DO NOTHING",
                (c, emb, f"{source}:{file.filename}", h),
            )
            if cur.rowcount == 1:
                inserted += 1
        conn.commit()
        cur.close()
        conn.close()
        return {"ok": True, "chunks_added": inserted, "source": f"{source}:{file.filename}"}
    except Exception as e:
        print("UPLOAD FAILED:", repr(e))
        raise HTTPException(status_code=500, detail=f"upload_failed: {repr(e)}")


# ─────────────────────────────────────────────────────────────
# Build LLM Messages
# 3-layer retrieval: Chroma → F5 Module Web Fallback → General Web
# ─────────────────────────────────────────────────────────────
def build_llm_messages(session_id: str, user_message: str) -> Dict[str, Any]:
    history_llm = get_history_for_llm(session_id, limit=8)
    module_hint = infer_module_hint(user_message)

    f5_context = ""
    f5_citations: List[Dict[str, Any]] = []
    f5_route = ""
    f5_ops_authoritative = False
    command_intent = is_command_intent(user_message)
    mutating_intent = is_mutating_command_request(user_message)

    # ── Layer 1: Chroma F5 Docs ───────────────────────────────
    if is_f5_query(user_message):
        try:
            f5 = search_f5_docs(user_message, top_k=10, module_hint=module_hint)
            f5_context = (f5.get("context") or "")[:6000]
            f5_citations = f5.get("citations", []) or []
            f5_route = (f5.get("route") or "")
            f5_ops_authoritative = has_authoritative_ops_evidence_for_module(
                f5_citations, module_hint
            )
            print(f"📘 Chroma [{module_hint.upper()}] returned {len(f5_context)} chars")
        except Exception as e:
            print(f"F5 docs retrieval error: {e}")

    # ── Command planner ───────────────────────────────────────
    planner_context = ""
    if is_f5_query(user_message) and command_intent:
        chosen = choose_template(user_message)
        if chosen:
            extracted = extract_params(user_message)
            miss = missing_fields(chosen, extracted)
            planner_context = (
                "### Command Planner (server-side)\n"
                f"- module_hint: {module_hint}\n"
                f"- template: {chosen.key}\n"
                f"- object_type: {chosen.object_type}\n"
                f"- action: {chosen.action}\n"
                f"- extracted: {extracted}\n"
                f"- missing_required: {miss}\n"
                "Rules:\n"
                "- If missing_required is not empty, DO NOT produce create/modify/delete commands.\n"
                "- Ask the user for exactly the missing fields.\n"
                "- If missing_required is empty, render tmsh using the template skeleton.\n"
            )

    # ── Layer 2: Postgres RAG ─────────────────────────────────
    rag_context = ""
    if should_use_rag(user_message):
        rag = search_knowledge_base(user_message, top_k=3, min_score=0.35)
        rag_context = "\n".join(rag) if rag else ""

    # ── Layer 3A: General web search (news, CVEs, time-sensitive)
    web_context = ""
    f5_web_fallback_used = False

    if needs_web_search(user_message):
        print(f"🔍 General web search triggered: {user_message[:80]}")
        web_context = web_search(user_message)

    # ── Layer 3B: F5 module-aware web fallback ────────────────
    # Fires when Chroma returned thin context for ANY F5 module
    # Covers: LTM, GTM, APM, WAF, AFM, SSLO, F5OS, Automation
    if not web_context and needs_f5_web_fallback(user_message, f5_context):
        print(f"🔍 F5 [{module_hint.upper()}] web fallback triggered: {user_message[:80]}")
        web_context = web_search_f5_targeted(user_message, module_hint=module_hint)
        f5_web_fallback_used = bool(web_context)

    sys_prompt = build_system_prompt(
        f5_context, rag_context, web_context, planner_context,
        module_hint=module_hint,
        f5_web_fallback_used=f5_web_fallback_used,
    )
    messages = [{"role": "system", "content": sys_prompt}] + history_llm

    command_blocked = bool(
        command_intent and mutating_intent
        and is_f5_query(user_message)
        and not f5_ops_authoritative
    )

    return {
        "messages":              messages,
        "module_hint":           module_hint,
        "rag_used":              bool(rag_context),
        "web_search_used":       bool(web_context),
        "f5_docs_used":          bool(f5_context),
        "f5_web_fallback_used":  f5_web_fallback_used,
        "f5_citations":          f5_citations,
        "f5_route":              f5_route,
        "command_intent":        bool(command_intent),
        "mutating_intent":       bool(mutating_intent),
        "f5_ops_authoritative":  bool(f5_ops_authoritative),
        "command_blocked":       command_blocked,
    }


# ─────────────────────────────────────────────────────────────
# SSE Helper
# ─────────────────────────────────────────────────────────────
def sse_event(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"



# ─────────────────────────────────────────────────────────────
# Troubleshooting detection + internal plan (to improve depth)
# ─────────────────────────────────────────────────────────────
TROUBLESHOOTING_HINTS = (
    "troubleshoot", "troubleshooting", "vip", "virtual", "virtual server", "pool down",
    "timeout", "timed out", "reset", "rst", "503", "handshake", "tls", "ssl", "intermittent",
    "connection refused", "no members available", "health monitor", "snat", "asymmetric"
)

def is_troubleshooting_query(q: str) -> bool:
    s = (q or "").lower()
    return any(h in s for h in TROUBLESHOOTING_HINTS)

def build_internal_troubleshooting_plan(user_q: str, module_hint: str) -> str:
    # Keep it short; never show to the user verbatim.
    return (
        "Plan (do not reveal):\n"
        "• Classify symptom (timeout vs reset vs 503 vs TLS fail).\n"
        "• Isolate layers: Client→F5 ingress→F5 selection→F5 egress→Server→Return path.\n"
        "• For each layer, propose 1-2 verification steps/commands (read-only).\n"
        "• List likely root causes ranked by probability for " + module_hint.upper() + "."
    )

# ─────────────────────────────────────────────────────────────
# Chat Endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    ensure_session_exists(session_id)

    for m in request.messages:
        save_message(session_id, m.role, m.content)

    touch_session(session_id)
    maybe_autotitle_session(session_id)

    user_message = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        request.messages[-1].content,
    )

    ctx = build_llm_messages(session_id, user_message)
    messages    = ctx["messages"]
    f5_citations = ctx.get("f5_citations", [])
    module_hint  = ctx.get("module_hint", "ltm")

    # ── Improve depth for troubleshooting questions ──
    if is_troubleshooting_query(user_message):
        plan = build_internal_troubleshooting_plan(user_message, module_hint)
        messages = messages + [{
            "role": "system",
            "content": plan + "\n\nUse this plan to produce the final response. Do NOT reveal or reference the plan."
        }]

    def meta_payload(extra: dict = {}) -> dict:
        return {
            "type":                  "meta",
            "rag_used":              ctx["rag_used"],
            "web_search_used":       ctx["web_search_used"],
            "f5_docs_used":          ctx["f5_docs_used"],
            "f5_web_fallback_used":  ctx.get("f5_web_fallback_used", False),
            "module_hint":           module_hint,
            "session_id":            session_id,
            **extra,
        }

    # ── Guard: block mutating tmsh without authoritative evidence ──
    if ctx.get("command_blocked"):
        reply = build_command_block_response(module_hint)
        save_message(session_id, "assistant", reply)
        touch_session(session_id)

        if request.stream:
            def gen_blocked() -> Generator[str, None, None]:
                yield sse_event(meta_payload({"blocked": True, "reason": "missing_authoritative_ops_evidence"}))
                if f5_citations:
                    yield sse_event({"type": "citations", "items": f5_citations})
                yield sse_event({"type": "delta", "text": reply})
                yield sse_event({"type": "done"})
            return StreamingResponse(
                gen_blocked(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )

        return {
            "reply": reply, "model": "guard/authoritative",
            "session_id": session_id, "module_hint": module_hint,
            "rag_used": ctx["rag_used"], "web_search_used": ctx["web_search_used"],
            "f5_docs_used": ctx["f5_docs_used"],
            "f5_web_fallback_used": ctx.get("f5_web_fallback_used", False),
            "citations": f5_citations, "blocked": True,
            "block_reason": "missing_authoritative_ops_evidence",
        }

    # ── Local Ollama ──────────────────────────────────────────
    if request.use_local:
        reply = await get_ollama_response(messages) or "Vittu Error: Local model failed."
        save_message(session_id, "assistant", reply)
        touch_session(session_id)
        return {
            "reply": reply, "model": "ollama/vittu",
            "session_id": session_id, "module_hint": module_hint,
            "rag_used": ctx["rag_used"], "web_search_used": ctx["web_search_used"],
            "f5_docs_used": ctx["f5_docs_used"],
            "f5_web_fallback_used": ctx.get("f5_web_fallback_used", False),
            "citations": f5_citations,
        }

    # ── Streaming SSE (Groq) ──────────────────────────────────
    if request.stream:
        def gen() -> Generator[str, None, None]:
            full = ""
            yield sse_event(meta_payload())
            if f5_citations:
                yield sse_event({"type": "citations", "items": f5_citations})
            try:
                stream = groq_client.chat.completions.create(
                    model=GROQ_MODEL_DEFAULT,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                    stream=True,
                )
                buf = ""
                last_flush = time.time()
                FLUSH_EVERY_CHARS = 80
                FLUSH_EVERY_SECS  = 0.08

                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta.content or ""
                    except Exception:
                        delta = ""
                    if not delta:
                        continue
                    full += delta
                    buf  += delta
                    now = time.time()
                    if len(buf) >= FLUSH_EVERY_CHARS or (now - last_flush) >= FLUSH_EVERY_SECS:
                        yield sse_event({"type": "delta", "text": buf})
                        buf = ""
                        last_flush = now

                if buf:
                    yield sse_event({"type": "delta", "text": buf})

                if ctx.get("command_intent") and not tmsh_output_looks_valid(full):
                    full = (
                        "I'm not going to output tmsh because the generated command format is not valid.\n\n"
                        "Reply with: object type + names + destination/members + profiles + SNAT/persistence.\n"
                        "Or ask for 'show/list' verification commands."
                    )

                save_message(session_id, "assistant", full)
                touch_session(session_id)
                yield sse_event({"type": "done"})

            except Exception as e:
                yield sse_event({"type": "error", "error": str(e)})

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # ── Non-streaming (Groq) ──────────────────────────────────
    groq_response = groq_client.chat.completions.create(
        model=GROQ_MODEL_DEFAULT,
        messages=messages,
        max_tokens=2048,
        temperature=0.3,
    )
    reply = groq_response.choices[0].message.content

    if ctx.get("command_intent") and not tmsh_output_looks_valid(reply):
        reply = (
            "I'm not going to output tmsh because the generated command format is not valid.\n\n"
            "Reply with: object type + names + destination/members + profiles + SNAT/persistence.\n"
            "Or ask for 'show/list' verification commands."
        )

    save_message(session_id, "assistant", reply)
    touch_session(session_id)

    return {
        "reply":                reply,
        "model":                f"groq/{GROQ_MODEL_DEFAULT}",
        "session_id":           session_id,
        "module_hint":          module_hint,
        "rag_used":             ctx["rag_used"],
        "web_search_used":      ctx["web_search_used"],
        "f5_docs_used":         ctx["f5_docs_used"],
        "f5_web_fallback_used": ctx.get("f5_web_fallback_used", False),
        "citations":            f5_citations,
    }


# ─────────────────────────────────────────────────────────────
# Health + Root
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    now = datetime.now()
    return {
        "status":            "ok",
        "date":              now.strftime("%A, %B %d, %Y"),
        "time":              now.strftime("%I:%M %p"),
        "groq_model":        GROQ_MODEL_DEFAULT,
        "local_model":       "ollama/vittu",
        "search":            "tavily",
        "pgvector_available": PGVECTOR_OK,
        "features": [
            "chroma_f5_docs",
            "postgres_rag",
            "tavily_general_web_search",
            "f5_module_aware_web_fallback",
            "ltm_gtm_apm_waf_afm_sslo_f5os_automation",
            "perplexity_style_citations",
            "authoritative_command_guard",
            "command_planner_templates",
            "module_hint_router",
            "source_transparency",
            "streaming_sse",
            "sessions_sidebar",
            "file_upload_rag",
            "local_ollama_fallback",
        ],
    }


@app.get("/")
async def root():
    return {"message": "Vittu AI Assistant is running 🚀"}
