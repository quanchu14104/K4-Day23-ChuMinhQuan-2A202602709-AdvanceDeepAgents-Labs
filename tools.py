"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

# Module-level variable to enforce arXiv 3-second etiquette
_last_arxiv_time = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError or retryable HTTP/network errors, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            retryable = False
            retry_after = None

            if isinstance(exc, RetryableError):
                retryable = True
                retry_after = exc.retry_after
            elif isinstance(exc, httpx.HTTPStatusError):
                if exc.response.status_code in {429, 500, 502, 503, 504}:
                    retryable = True
                    ra_hdr = exc.response.headers.get("Retry-After")
                    if ra_hdr:
                        try:
                            retry_after = float(ra_hdr)
                        except ValueError:
                            pass
            elif isinstance(exc, httpx.TransportError):
                retryable = True

            if not retryable:
                raise exc

            if attempt == attempts - 1:
                raise exc

            if retry_after is not None and retry_after > 0:
                delay = min(float(retry_after), cap)
            else:
                backoff = base * (2 ** attempt)
                jitter = random.uniform(0.1, 0.5 * min(backoff, cap))
                delay = min(cap, backoff + jitter)

            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _last_arxiv_time
    try:
        terms = re.findall(r"[\w-]+", query)
        if not terms:
            return "NO RESULTS"

        now = time.monotonic()
        wait_seconds = 3.0 - (now - _last_arxiv_time)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _last_arxiv_time = time.monotonic()

        search_query = " AND ".join(f"all:{t}" for t in terms)
        clamped_max = max(1, min(int(max_results), 30))
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": clamped_max,
        }

        def _fetch():
            resp = httpx.get(ARXIV_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp

        resp = with_retry(_fetch, attempts=5, base=2.0, cap=60.0)

        root = xml.etree.ElementTree.fromstring(resp.content)
        atom_ns = "{http://www.w3.org/2005/Atom}"
        entries = root.findall(f"{atom_ns}entry")
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            id_el = entry.find(f"{atom_ns}id")
            pub_el = entry.find(f"{atom_ns}published")
            title_el = entry.find(f"{atom_ns}title")
            summary_el = entry.find(f"{atom_ns}summary")

            raw_id = (id_el.text or "").strip()
            if "/abs/" in raw_id:
                raw_id = raw_id.split("/abs/")[-1]
            else:
                raw_id = raw_id.split("/")[-1]
            clean_id = re.sub(r"v\d+$", "", raw_id)
            url = f"https://arxiv.org/abs/{clean_id}"

            published = (pub_el.text or "").strip()[:10] if pub_el is not None else ""
            title = " ".join((title_el.text or "").split()) if title_el is not None else ""
            summary = " ".join((summary_el.text or "").split())[:600] if summary_el is not None else ""

            records.append({
                "id": clean_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(int(limit), 100))
        params = {"limit": clamped_limit}
        if date and date.strip():
            params["date"] = date.strip()

        def _fetch():
            resp = httpx.get(HF_DAILY_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp

        resp = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        items = resp.json()
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        kw = keyword.strip().lower() if keyword else ""
        for item in items:
            paper = item.get("paper") or {}
            paper_id = paper.get("id")
            if not paper_id:
                continue

            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = " ".join(str(paper.get("title") or item.get("title") or "").split())
            summary = " ".join(str(paper.get("summary") or item.get("summary") or "").split())[:600]
            upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
            github = str(paper.get("githubRepo") or item.get("githubRepo") or "")
            stars = int(paper.get("githubStars") or item.get("githubStars") or 0)
            url = f"https://huggingface.co/papers/{paper_id}"

            if kw and (kw not in title.lower() and kw not in summary.lower()):
                continue

            records.append({
                "id": str(paper_id),
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        clamped_limit = max(1, min(int(limit), 50))
        params = {"q": query, "limit": clamped_limit}

        def _fetch():
            resp = httpx.get(HF_SEARCH_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp

        resp = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        items = resp.json()
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        for item in items:
            paper = item.get("paper") if isinstance(item, dict) and "paper" in item else item
            if not isinstance(paper, dict):
                continue
            paper_id = paper.get("id")
            if not paper_id:
                continue

            published = str(paper.get("publishedAt") or "")[:10]
            title = " ".join(str(paper.get("title") or "").split())
            raw_summary = paper.get("ai_summary") or paper.get("summary") or ""
            summary = " ".join(str(raw_summary).split())[:600]
            upvotes = int(paper.get("upvotes") or 0)
            github = str(paper.get("githubRepo") or "")
            stars = int(paper.get("githubStars") or 0)
            url = f"https://huggingface.co/papers/{paper_id}"

            records.append({
                "id": str(paper_id),
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _redact_key(text: str) -> str:
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key and key in text:
        return text.replace(key, "[REDACTED]")
    return text


def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    key = (os.getenv("EXA_API_KEY") or "").strip()
    endpoint = f"{EXA_URL}?exaApiKey={key}" if key else EXA_URL
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    def _request():
        resp = httpx.post(endpoint, json=payload, headers=headers, timeout=30.0)
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            delay = float(ra) if ra else 20.0
            raise RetryableError("Exa HTTP 429 Rate Limited", retry_after=delay)
        resp.raise_for_status()

        raw_text = resp.text
        data = None
        for line in raw_text.splitlines():
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                    break
                except json.JSONDecodeError:
                    continue
        if data is None:
            try:
                data = resp.json()
            except Exception:
                raise ValueError(f"Invalid Exa MCP response format: {raw_text[:200]}")

        if "error" in data:
            err_str = str(data["error"])
            if any(k in err_str.lower() for k in ["rate limit", "quota", "too many requests"]):
                raise RetryableError(f"Exa rate limit: {err_str}", retry_after=20.0)
            raise RuntimeError(f"Exa JSON-RPC error: {err_str}")

        result = data.get("result", {})
        meta = result.get("_meta", {})
        meta_str = str(meta).lower()
        if "rate limit" in meta_str or "quota" in meta_str:
            raise RetryableError("Exa rate limited in metadata", retry_after=20.0)

        contents = result.get("content", [])
        texts = []
        for item in contents:
            if item.get("type") == "text":
                t = item.get("text", "")
                if any(term in t.lower() for term in ["rate limit exceeded", "too many requests", "free tier rate limit"]):
                    raise RetryableError("Exa rate limit message in content", retry_after=20.0)
                texts.append(t)
        return "\n\n".join(texts).strip()

    return with_retry(_request, attempts=5, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        num = max(1, min(int(num_results), 20))
        obj = objective.strip() if objective and objective.strip() else f"Find authoritative research papers and surveys on {query}"
        res = _call_exa_mcp("web_search_exa", {"query": query, "objective": obj, "numResults": num})
        return res if res else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact_key(str(exc))}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        res = _call_exa_mcp("web_fetch_exa", {"urls": [url]})
        if not res:
            return "NO RESULTS"
        return res[:12000]
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact_key(str(exc))}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
