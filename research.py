"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    s = str(topic or "").strip().lower()
    s = re.sub(r"[^\w]+", "-", s).strip("-")
    if len(s) > 60:
        s = s[:60].rstrip("-")
    return s if s else "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return f"""Please perform in-depth multi-source research and write a comprehensive literature review and survey report on:
"{topic}"

You must execute the entire investigation autonomously to completion without pausing for user input:
1. First, call `write_todos` to lay out the plan with at least 3 sub-questions.
2. Immediately call the `task` tool to delegate each sub-question to the `researcher` subagent:
   - Delegation 1: instruct researcher to use `arxiv_search` AND `hf_search_papers`.
   - Delegation 2: instruct researcher to use `hf_daily_papers` (or `hf_search_papers`) AND `web_search`.
   - Delegation 3: instruct researcher to use `arxiv_search` AND `web_search`.
   Ensure target notes files are in /tmp/work/research/notes/ (e.g. 01-slug.md, 02-slug.md, 03-slug.md).
3. Check note files with `read_file` or `ls`, and compile /tmp/work/research/sources.json ensuring at least 3 distinct source families (arxiv, hf-daily, hf-search, web) are present.
4. Write the comprehensive report body into /tmp/work/report/report.md with inline citations [n] from all available source families. (Do NOT write ## References).
5. Run `execute` with command: python3 /tmp/work/research/finalize_citations.py
6. Run `execute` with command: python3 /tmp/work/research/check_citations.py until it prints OK.
7. Delegate 2 claims to the `citation-checker` subagent using `task`.
Begin now by calling write_todos and task!
"""


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_counter = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        t_calls = getattr(msg, "tool_calls", None)
        if t_calls is None and isinstance(msg, dict):
            t_calls = msg.get("tool_calls")
        if t_calls:
            for call in t_calls:
                call_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if call_name:
                    tool_counter[call_name] += 1

        usage = getattr(msg, "usage_metadata", None)
        if usage is None and isinstance(msg, dict):
            usage = msg.get("usage_metadata")
        if usage and isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)

    subagent_calls = tool_counter.get("task", 0)
    return {
        "model": str(model_name),
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_counter),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    slug = slugify(topic)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"Report is missing or empty at {REPORT_PATH}")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError(f"sources.json is missing or empty at {SOURCES_PATH}")

    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources_data, list):
            raise ValueError("sources.json is not a list")
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}")

    summary_info = summarize(messages, elapsed, model_name)
    distinct_families = sorted({str(s.get("source")) for s in sources_data if s.get("source")})

    meta_data = {
        "topic": topic,
        **summary_info,
        "n_sources": len(sources_data),
        "source_families": distinct_families,
    }

    report_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"

    sources_path.write_text(json.dumps(sources_data, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report_bytes.decode("utf-8"), encoding="utf-8")

    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    if not topic or not topic.strip():
        print("Usage: python research.py '<topic>'", file=sys.stderr)
        return 2

    topic = topic.strip()
    model = make_model()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "model")

    start = time.monotonic()
    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
        })
        agent = build_lead_agent(backend, model)
        user_prompt = build_prompt(topic)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_prompt}]},
            config={"recursion_limit": 1000},
        )
        elapsed = time.monotonic() - start
        messages = result.get("messages", []) if isinstance(result, dict) else []
        print(f"--- AGENT FINISHED in {elapsed:.1f}s with {len(messages)} messages ---")
        for idx, msg in enumerate(messages):
            role = getattr(msg, "type", type(msg).__name__)
            t_calls = getattr(msg, "tool_calls", None) or []
            call_names = [c.get("name") for c in t_calls] if t_calls else []
            content_preview = str(getattr(msg, "content", ""))[:120].replace("\n", " ")
            print(f"[{idx}] {role}: {content_preview}")
            if call_names:
                print(f"    Tools called: {call_names}")
        try:
            saved_report = save_outputs(backend, topic, messages, elapsed, model_name)
            print(f"SUCCESS: Report saved to {saved_report}")
            return 0
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
