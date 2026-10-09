"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
_REF_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$")
_URL = re.compile(r"https?://[^\s)\]>]+")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            try:
                numbers.append(int(part))
            except ValueError:
                pass
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not sources or not isinstance(sources, list):
        return ["no sources in sources.json"]

    seen_urls = {}
    source_by_n = {}
    for idx, s in enumerate(sources):
        if not isinstance(s, dict):
            problems.append(f"source item at index {idx} is not an object")
            continue
        n = s.get("n")
        if not isinstance(n, int):
            problems.append(f"source {s.get('id', idx)} has non-integer n={n!r}")
            continue
        url = str(s.get("url") or "").strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] url must start with http:// or https://: {url!r}")
        if url in seen_urls:
            problems.append(f"duplicate url {url} in sources [{seen_urls[url]}] and [{n}]")
        else:
            seen_urls[url] = n
        if n in source_by_n:
            problems.append(f"source number [{n}] defined multiple times in sources.json")
        else:
            source_by_n[n] = s

    # Split report_text at ## References
    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing '## References' heading in report")
        body = report_text
        ref_text = ""
    else:
        last_match = matches[-1]
        body = report_text[:last_match.start()]
        ref_text = report_text[last_match.end():]

    # Find cited in body (ignoring code blocks and markdown links [n](url))
    segments = _CODE.split(body)
    cited = set()
    for i, seg in enumerate(segments):
        if i % 2:  # inside code block / inline code
            continue
        for m in _GROUP.finditer(seg):
            for num in _group_numbers(m.group(1)):
                cited.add(num)

    # Check cited against sources
    for n in sorted(cited):
        if n not in source_by_n:
            problems.append(f"[{n}] cited in text but missing from sources.json")

    for n in sorted(source_by_n.keys()):
        if n not in cited:
            problems.append(f"source [{n}] never cited")

    # Check ## References lines
    ref_lines = {}
    for line in ref_text.splitlines():
        line = line.strip()
        if not line:
            continue
        rm = _REF_LINE.match(line)
        if rm:
            num = int(rm.group(1))
            line_content = line
            if num in ref_lines:
                problems.append(f"multiple reference lines for [{num}] in ## References")
            else:
                ref_lines[num] = line_content

    # Every source needs exactly one reference line
    for n in sorted(source_by_n.keys()):
        if n not in ref_lines:
            problems.append(f"missing reference line for source [{n}] in ## References")
        else:
            line_str = ref_lines[n]
            raw_urls = _URL.findall(line_str)
            cleaned_urls = [u.rstrip(").,]>") for u in raw_urls]
            if len(cleaned_urls) != 1:
                problems.append(f"reference line [{n}] must contain exactly one URL, found {len(cleaned_urls)}")
            elif source_by_n[n].get("url"):
                expected_url = source_by_n[n]["url"].rstrip(").,]>")
                if cleaned_urls[0] != expected_url:
                    problems.append(f"reference line [{n}] URL '{cleaned_urls[0]}' does not match source URL '{expected_url}'")

    for num in sorted(ref_lines.keys()):
        if num not in source_by_n:
            problems.append(f"reference line [{num}] is not in sources.json")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
