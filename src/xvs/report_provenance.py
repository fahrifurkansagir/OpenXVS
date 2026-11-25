"""Report Provenance Tracking Infrastructure (W5 / Charter §8 / G-B14 Step 2).

Provides:
- rv(source, source_path, key_path, fmt): Retrieves a value from a loaded JSON/config/manifest
  dict or list, formats it, records its source, key_path, raw_value, and formatted value in the
  active report's provenance tracker, and returns the formatted string.
  Raises TypeError on bare numeric, boolean, or string literals.
- register_json_source(obj, source_path): Recursively tags a loaded JSON dict/list hierarchy with
  its relative source file path and dotted key prefix so nested dicts passed to rv() retain full
  (source, key_path) provenance.
- load_json_source(path, source_label): Loads a JSON file and registers its provenance tree.
- ProvenanceTracker: Context manager and container for recording all values inserted into a report.
- save_provenance(report_path): Serializes recorded entries and positional bindings for all table/prose
  numbers and PASS/FAIL verdicts to <report_path>.provenance.json.
- verify_report_provenance(report_path): Verifies every non-excepted number and PASS/FAIL verdict against
  its recorded provenance entry (source, key_path, formatted value, and position).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

_JSON_CACHE: dict[str, Any] = {}
_ACTIVE_TRACKER: ProvenanceTracker | None = None
_LAST_TRACKER: ProvenanceTracker | None = None

# Maps id(dict_or_list) -> (source_path_str, dotted_key_prefix)
_OBJECT_SOURCES: dict[int, tuple[str, str]] = {}
# Keep references alive so Python does not GC and reuse object IDs while tracking
_OBJECT_KEEPALIVE: list[Any] = []

_NUMBER_REGEX = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])"
)
_VERDICT_REGEX = re.compile(r"\b(PASS|FAIL)\b")

_FORMAT_SPEC_REGEX = re.compile(
    r"^(?:pct:|div60:|div3600:|mul:[-+]?\d+(?:\.\d+)?:)?(?:[<>^=+ -]?[#0]?[0-9]*[_,]?(?:\.[0-9]+)?[bcdeEfFgGnosxX%]?|argv_ps1|verdict)$"
)


def _normalize_source_label(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        s = p.as_posix()
        for marker in ("output/xvs/", "configs/xvs/", "docs/xvs/", "tests/"):
            idx = s.find(marker)
            if idx != -1:
                return s[idx:]
        return p.name if p.name else str(path)


def register_json_source(obj: Any, source_path: str | Path, prefix: str = "") -> Any:
    """Recursively register a loaded JSON/config/manifest dict or list with its source path and key prefix."""
    src_str = _normalize_source_label(source_path) if isinstance(source_path, Path) or ("/" in str(source_path) or "\\" in str(source_path)) else str(source_path)
    _OBJECT_KEEPALIVE.append(obj)

    def _walk(node: Any, cur_prefix: str) -> None:
        if isinstance(node, dict):
            _OBJECT_SOURCES[id(node)] = (src_str, cur_prefix)
            for k, v in node.items():
                next_prefix = f"{cur_prefix}.{k}" if cur_prefix else str(k)
                if isinstance(v, (dict, list)):
                    _walk(v, next_prefix)
        elif isinstance(node, list):
            _OBJECT_SOURCES[id(node)] = (src_str, cur_prefix)
            for idx, v in enumerate(node):
                next_prefix = f"{cur_prefix}.{idx}" if cur_prefix else str(idx)
                if isinstance(v, (dict, list)):
                    _walk(v, next_prefix)

    _walk(obj, prefix)
    return obj


def load_json_source(path: Path | str, source_label: str | None = None) -> Any:
    """Load a JSON file from disk and register all nested dicts/lists for provenance tracking."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    label = source_label if source_label is not None else _normalize_source_label(p)
    return register_json_source(data, label)


def resolve_key_path(obj: Any, key_path: str) -> Any:
    """Traverse a key path in nested dicts/lists.

    Supports dot-notation and handles keys containing dots (e.g. '100.0', '0.01').
    """
    if key_path == "" or key_path is None:
        return obj

    if isinstance(obj, dict) and key_path in obj:
        return obj[key_path]

    parts = key_path.split(".")
    curr = obj
    i = 0
    while i < len(parts):
        part = parts[i]
        if isinstance(curr, dict):
            if part in curr:
                curr = curr[part]
                i += 1
            else:
                found = False
                for j in range(len(parts), i, -1):
                    combined = ".".join(parts[i:j])
                    if combined in curr:
                        curr = curr[combined]
                        i = j
                        found = True
                        break
                if not found:
                    raise KeyError(f"Key '{part}' not found in dict keys: {list(curr.keys())[:10]}")
        elif isinstance(curr, (list, tuple)):
            try:
                idx = int(part)
                curr = curr[idx]
                i += 1
            except (ValueError, IndexError) as err:
                raise KeyError(f"Invalid list index '{part}' for list of length {len(curr)}: {err}")
        else:
            raise KeyError(f"Cannot traverse key '{part}' on non-collection type {type(curr)}")
    return curr


def format_value(val: Any, fmt: str = "") -> str:
    """Format a value according to format string (supports pct:, div60:, div3600:, mul:, verdict, argv_ps1)."""
    if fmt == "verdict":
        if isinstance(val, bool):
            return "PASS" if val else "FAIL"
        return str(val)
    if fmt == "argv_ps1":
        if isinstance(val, (list, tuple)):
            parts = []
            for tok in val:
                s = str(tok).replace("\\", "/")
                if " " in s:
                    parts.append(f'"{s}"')
                else:
                    parts.append(s)
            return "& .venv/Scripts/python.exe " + " ".join(parts)
        return str(val)

    scale = 1.0
    sub_fmt = fmt
    if fmt.startswith("pct:"):
        scale = 100.0
        sub_fmt = fmt[4:]
    elif fmt.startswith("div60:"):
        scale = 1.0 / 60.0
        sub_fmt = fmt[6:]
    elif fmt.startswith("div3600:"):
        scale = 1.0 / 3600.0
        sub_fmt = fmt[8:]
    elif fmt.startswith("mul:"):
        rest = fmt[4:]
        sep_idx = rest.find(":")
        if sep_idx != -1:
            scale = float(rest[:sep_idx])
            sub_fmt = rest[sep_idx + 1 :]

    if scale != 1.0 and isinstance(val, (int, float)) and not isinstance(val, bool):
        val = float(val) * scale

    if not sub_fmt:
        return str(val)
    if "{" in sub_fmt and "}" in sub_fmt:
        return sub_fmt.format(val)
    try:
        return format(val, sub_fmt)
    except (ValueError, TypeError):
        return str(val)


def _get_exception_intervals(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (all_exception_intervals, test_output_intervals) for a report markdown text."""
    test_output_rx = re.compile(r"## 4\.\s*Test output[\s\S]*?```[a-zA-Z]*\r?\n[\s\S]*?\r?\n```")
    test_intervals = [(m.start(), m.end()) for m in test_output_rx.finditer(text)]

    exception_regexes = [
        test_output_rx,
        # Section and list numbering
        re.compile(r"^#+\s+(?:Table\s+)?\d+(?:\.\d+)*\b", re.MULTILINE),
        re.compile(r"[§#]\s*\d+(?:\.\d+)*"),
        re.compile(r"^\s*(?:[-*]\s+)?\d+\.\s+", re.MULTILINE),
        re.compile(r"^\s*#\s*\d+\.\s+.*$", re.MULTILINE),
        # Dates and times
        re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?\b"),
        re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
        # Hashes (commit hashes, sha256)
        re.compile(r"\b[0-9a-fA-F]{7,64}\b"),
        # Identifiers containing digits: WP-05, R-GB2-1, D27, V2, T1, C1_UAS, file names, math symbols
        re.compile(r"\b[A-Za-z][A-Za-z0-9_-]*\d+[A-Za-z0-9_-]*\b|\b\d+[A-Za-z]+[A-Za-z0-9_-]*\b"),
        re.compile(r"\b[A-Za-z]+-\d+[A-Za-z0-9_-]*\b"),
        re.compile(r"\$[A-Za-z0-9_{}^\\+-]+_\{?\d+[A-Za-z0-9]*\}?\$|\b[A-Za-z]_\{?\d+\}?|\$\s*1\s*-\s*\\alpha\s*\$"),
        re.compile(
            r"\b(?:EMVA\s*1288(?:\s*Release\s*\d+)?|DO-\d+(?:\s*(?:requirement|§)?\s*[\d.]+)?|ASTM\s*F\d+(?:-\d+)?|Rec\.\s*\d+|IEC\s*\d+(?:-\d+)*|LINE_AA)\b",
            re.IGNORECASE,
        ),
        re.compile(r"[\w.-]+[/\\][\w.-]+|[\w.-]+\.(?:py|json|jsonl|md|png|csv|pkl|gz|txt|sh|ipynb)"),
        re.compile(
            r"[~≈]?\b\d+-bit\b|[~≈]?\b\d+[x×]\d+\b|[~≈]?\b\d+\s*\\times\s*\d+\b|[~≈]?\b\d+\^2\b|\b\d+[²³]\b|\b\d+:\d+\b|\b\d+°(?:\s*[x×]\s*\d+°)?|\b\d+\^\\circ\b|[~≈]?\b\d+(?:\.\d+)?x\b|\b\d+-cam(?:era)?\b"
        ),
        re.compile(r"10\^\{?-?\d+\}?|\^[0-9]+"),
        # Specific table column headings and static CI descriptions
        re.compile(
            r"\b(?:95%|99%|%95)\s*(?:Confidence\s+Interval|CI|Exact\s+Upper\s+Bound|exact\s+Clopper-Pearson|bootstrap\s+intervals|Wald\s+CI|Clopper-Pearson|güven\s+aralıkları|Wald\s+95%)\b"
            r"|\|\s*\b95%\s+Confidence\s+Interval\b"
            r"|\|\s*\b95%\s+Exact\s+Upper\s+Bound\b"
            r"|\b\\pm\s*2\\%\s*Band\b"
            r"|Poisson\s*95%\s*Upper\s*Bound"
            r"|Poisson\s*95%\s*UB"
            r"|Wald\s*95%\s*CI"
            r"|Clopper-Pearson\s*95%\s*CIs?"
            r"|95%\s*Clopper-Pearson\s*CI"
            r"|95%\s*bootstrap\s*intervals"
            r"|%95\s*güven\s*aralıklarıyla",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:Item|Step|Phase|Table|Figure)\s*\d+\b", re.IGNORECASE),
        re.compile(r"<!--.*?-->", re.DOTALL),
    ]

    intervals: list[tuple[int, int]] = []
    for rx in exception_regexes:
        for m in rx.finditer(text):
            intervals.append((m.start(), m.end()))
    return intervals, test_intervals


def _extract_report_tokens(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract all non-excepted number occurrences and PASS/FAIL tokens with line/col positions."""
    intervals, test_intervals = _get_exception_intervals(text)

    def is_excepted(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in intervals)

    def is_in_test_output(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in test_intervals)

    line_starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(idx + 1)
    lines = text.splitlines()

    def offset_to_line_col(offset: int) -> tuple[int, int, bool]:
        # Binary search for line index
        lo, hi = 0, len(line_starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if line_starts[mid] <= offset:
                lo = mid + 1
            else:
                hi = mid - 1
        line_idx = max(0, hi)
        col = offset - line_starts[line_idx]
        line_str = lines[line_idx].lstrip() if line_idx < len(lines) else ""
        is_tbl = line_str.startswith("|")
        return line_idx + 1, col, is_tbl

    num_tokens: list[dict[str, Any]] = []
    for m in _NUMBER_REGEX.finditer(text):
        if is_excepted(m.start(), m.end()):
            continue
        line_no, col_no, is_tbl = offset_to_line_col(m.start())
        num_tokens.append({
            "value": m.group(),
            "line": line_no,
            "col": col_no,
            "is_table": is_tbl,
        })

    verdict_tokens: list[dict[str, Any]] = []
    for m in _VERDICT_REGEX.finditer(text):
        if is_in_test_output(m.start(), m.end()):
            continue
        line_no, col_no, is_tbl = offset_to_line_col(m.start())
        verdict_tokens.append({
            "verdict": m.group(1),
            "line": line_no,
            "col": col_no,
            "is_table": is_tbl,
        })

    return num_tokens, verdict_tokens


def _num_matches_set(n_str: str, candidate_set: set[str]) -> bool:
    n_clean = n_str.lstrip("+-").replace(",", "")
    if n_str in candidate_set or n_clean in candidate_set:
        return True
    try:
        f = float(n_clean)
        if f"{f:g}" in candidate_set or f"{f}" in candidate_set or (f.is_integer() and str(int(f)) in candidate_set):
            return True
    except ValueError:
        pass
    return False


class ProvenanceTracker:
    """Tracks values inserted into a report for provenance verification."""

    def __init__(self, report_path: Path | str | None = None) -> None:
        self.report_path = Path(report_path).resolve() if report_path else None
        self.entries: list[dict[str, Any]] = []
        self.numbers: set[str] = set()
        self._prev_active: ProvenanceTracker | None = None

    def reset(self) -> None:
        self.entries.clear()
        self.numbers.clear()

    def __enter__(self) -> ProvenanceTracker:
        global _ACTIVE_TRACKER
        self._prev_active = _ACTIVE_TRACKER
        _ACTIVE_TRACKER = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        global _ACTIVE_TRACKER, _LAST_TRACKER
        _LAST_TRACKER = self
        if self.report_path and exc_type is None:
            self.save(self.report_path)
        _ACTIVE_TRACKER = self._prev_active

    def record(self, source_name: str, key_path: str, raw_val: Any, formatted: str) -> None:
        entry_nums: set[str] = set()
        for text in (formatted, formatted.replace(",", "")):
            for m in _NUMBER_REGEX.finditer(text):
                num_str = m.group()
                entry_nums.add(num_str)
                entry_nums.add(num_str.lstrip("+-"))
                try:
                    fval = float(num_str)
                    entry_nums.add(f"{fval:g}")
                    entry_nums.add(f"{fval}")
                    if fval.is_integer():
                        entry_nums.add(str(int(fval)))
                except ValueError:
                    pass
        if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
            entry_nums.add(str(raw_val))
            entry_nums.add(f"{raw_val:g}")
            if float(raw_val).is_integer():
                entry_nums.add(str(int(raw_val)))

        self.numbers.update(entry_nums)
        self.entries.append({
            "source": str(source_name),
            "key_path": str(key_path),
            "raw_value": raw_val,
            "formatted": formatted,
            "entry_numbers": sorted(entry_nums),
        })

    def rv(
        self,
        source: Any,
        arg1: str | Path = "",
        arg2: str = "",
        arg3: str = "",
        *,
        source_path: str | Path | None = None,
    ) -> str:
        # Reject bare numeric, boolean, None, or non-JSON-path string literals (Step 2.1 / F3)
        if source is None or isinstance(source, (int, float, bool, complex)):
            raise TypeError(
                f"rv() does not accept bare numeric/boolean literals ({source!r}); "
                "pass (loaded_json_obj, source_path, key_path, fmt)"
            )
        if isinstance(source, str) and not (source.endswith(".json") or Path(source).is_file()):
            raise TypeError(
                f"rv() does not accept bare string literals ({source!r}); "
                "pass (loaded_json_obj, source_path, key_path, fmt)"
            )

        # Resolve (data_obj, resolved_source_path, key_path, fmt)
        if isinstance(source, (str, Path)) and (str(source).endswith(".json") or Path(source).is_file()):
            src_path = Path(source)
            path_key = str(src_path.resolve())
            if path_key not in _JSON_CACHE:
                if not src_path.is_file():
                    raise FileNotFoundError(f"Provenance source file not found: {src_path}")
                _JSON_CACHE[path_key] = load_json_source(src_path)
            data = _JSON_CACHE[path_key]
            resolved_source = _normalize_source_label(src_path)
            # Determine if arg1 is key_path or repeated source_path
            s_arg1 = str(arg1)
            if arg3 != "" or (s_arg1.endswith(".json") and arg2 != ""):
                resolved_source = _normalize_source_label(s_arg1)
                key_path = str(arg2)
                fmt = str(arg3)
            else:
                key_path = s_arg1
                fmt = str(arg2)
        elif isinstance(source, (dict, list)):
            data = source
            reg_info = _OBJECT_SOURCES.get(id(source))
            s_arg1 = str(arg1)
            # Determine whether call is 4-arg (obj, source_path, key_path, fmt) or 3-arg (obj, key_path, fmt)
            is_4arg = False
            if arg3 != "":
                is_4arg = True
            elif source_path is not None:
                is_4arg = False
            elif s_arg1.endswith(".json") or "/" in s_arg1 or "\\" in s_arg1:
                # Check if s_arg1 is actually a key inside source
                if not (isinstance(source, dict) and s_arg1 in source) and arg2 != "":
                    is_4arg = True
            elif reg_info is None and arg2 != "" and not _FORMAT_SPEC_REGEX.match(arg2):
                is_4arg = True

            if is_4arg:
                resolved_source = _normalize_source_label(s_arg1) if ("/" in s_arg1 or "\\" in s_arg1) else s_arg1
                prefix = reg_info[1] if reg_info is not None else ""
                key_path = str(arg2)
                fmt = str(arg3)
                if reg_info is None:
                    register_json_source(source, resolved_source)
            else:
                if source_path is not None:
                    resolved_source = _normalize_source_label(source_path)
                    prefix = reg_info[1] if reg_info is not None else ""
                elif reg_info is not None:
                    resolved_source, prefix = reg_info
                else:
                    raise TypeError(
                        "rv() requires a source_path for unregistered dict/list objects: "
                        "pass rv(obj, source_path, key_path, fmt) or load via load_json_source()"
                    )
                key_path = s_arg1
                fmt = str(arg2)
        else:
            raise TypeError(
                f"rv() requires a loaded JSON/config/manifest dict or list, got {type(source).__name__}"
            )

        if key_path == "" or key_path is None:
            raise ValueError("rv() requires a non-empty key_path")

        raw_val = resolve_key_path(data, key_path)
        reg_info_now = _OBJECT_SOURCES.get(id(data))
        prefix_now = reg_info_now[1] if reg_info_now is not None else ""
        full_key_path = f"{prefix_now}.{key_path}" if prefix_now else str(key_path)

        formatted = format_value(raw_val, fmt)
        self.record(resolved_source, full_key_path, raw_val, formatted)
        return formatted

    def save(self, report_path: Path | str | None = None) -> Path:
        out_path = Path(report_path).resolve() if report_path else self.report_path
        if not out_path:
            raise ValueError("No report_path specified for provenance saving")

        clean_entries = []
        entry_num_sets: list[set[str]] = []
        for idx, e in enumerate(self.entries):
            entry_num_sets.append(set(e.get("entry_numbers", [])))
            clean_entries.append({
                "index": idx,
                "source": e["source"],
                "key_path": e["key_path"],
                "raw_value": e["raw_value"],
                "formatted": e["formatted"],
            })

        number_positions: list[dict[str, Any]] = []
        verdict_positions: list[dict[str, Any]] = []

        if out_path.is_file():
            text = out_path.read_text(encoding="utf-8")
            num_tokens, v_tokens = _extract_report_tokens(text)

            # Bind each extracted number occurrence in the report to its provenance entry and position
            search_cursor = 0
            n_entries = len(clean_entries)
            for tok in num_tokens:
                val_str = tok["value"]
                matched_idx = None
                # Prefer forward sequential match from search_cursor
                for offset in range(n_entries):
                    cand_idx = (search_cursor + offset) % max(1, n_entries)
                    if _num_matches_set(val_str, entry_num_sets[cand_idx]):
                        matched_idx = cand_idx
                        search_cursor = cand_idx
                        break
                if matched_idx is not None:
                    ent = clean_entries[matched_idx]
                    number_positions.append({
                        "line": tok["line"],
                        "col": tok["col"],
                        "is_table": tok["is_table"],
                        "value": val_str,
                        "formatted": ent["formatted"],
                        "source": ent["source"],
                        "key_path": ent["key_path"],
                        "entry_index": matched_idx,
                    })
                else:
                    number_positions.append({
                        "line": tok["line"],
                        "col": tok["col"],
                        "is_table": tok["is_table"],
                        "value": val_str,
                        "formatted": None,
                        "source": None,
                        "key_path": None,
                        "entry_index": None,
                    })

            # Bind each PASS/FAIL token in the report to its position and provenance entry (if any)
            v_entries = [
                e for e in clean_entries
                if isinstance(e["formatted"], str) and ("PASS" in e["formatted"] or "FAIL" in e["formatted"])
            ]
            v_cursor = 0
            for vtok in v_tokens:
                v_str = vtok["verdict"]
                v_src = "report_verdict"
                v_key = f"line_{vtok['line']}_col_{vtok['col']}"
                for offset in range(len(v_entries)):
                    cand = v_entries[(v_cursor + offset) % len(v_entries)]
                    if v_str in str(cand["formatted"]):
                        v_src = cand["source"]
                        v_key = cand["key_path"]
                        v_cursor = (v_cursor + offset + 1) % max(1, len(v_entries))
                        break
                verdict_positions.append({
                    "line": vtok["line"],
                    "col": vtok["col"],
                    "is_table": vtok["is_table"],
                    "verdict": v_str,
                    "source": v_src,
                    "key_path": v_key,
                })

        prov_path = out_path.parent / f"{out_path.name}.provenance.json"
        payload = {
            "report": out_path.as_posix(),
            "count": len(clean_entries),
            "numbers": sorted(list(self.numbers)),
            "entries": clean_entries,
            "number_positions": number_positions,
            "verdict_positions": verdict_positions,
        }
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return prov_path


ReportProvenanceTracker = ProvenanceTracker


def get_active_tracker() -> ProvenanceTracker:
    global _ACTIVE_TRACKER
    if _ACTIVE_TRACKER is None:
        _ACTIVE_TRACKER = ProvenanceTracker()
    return _ACTIVE_TRACKER


def set_active_tracker(tracker: ProvenanceTracker | None) -> None:
    global _ACTIVE_TRACKER
    _ACTIVE_TRACKER = tracker


def rv(
    source: Any,
    arg1: str | Path = "",
    arg2: str = "",
    arg3: str = "",
    *,
    source_path: str | Path | None = None,
) -> str:
    """Insert a provenanced value from a loaded JSON/config/manifest object into a report."""
    tracker = get_active_tracker()
    return tracker.rv(source, arg1, arg2, arg3, source_path=source_path)


def save_provenance(report_path: Path | str) -> Path:
    """Save the active provenance tracker and positional report bindings to <report_path>.provenance.json."""
    tracker = _ACTIVE_TRACKER or _LAST_TRACKER or get_active_tracker()
    return tracker.save(report_path)


def verify_report_provenance(report_path: Path | str, prov_path: Path | str | None = None) -> list[str]:
    """Verify that every number and PASS/FAIL verdict in a report matches its provenance entry and position.

    Returns a list of unprovenanced or position-mismatched tokens found. Empty list means PASS.
    """
    rep_p = Path(report_path).resolve()
    if prov_path is None:
        prov_p = rep_p.parent / f"{rep_p.name}.provenance.json"
    else:
        prov_p = Path(prov_path).resolve()

    if not prov_p.is_file():
        raise FileNotFoundError(f"Provenance file not found: {prov_p}")

    with open(prov_p, "r", encoding="utf-8") as f:
        prov_data = json.load(f)

    valid_numbers = set(prov_data.get("numbers", []))

    with open(rep_p, "r", encoding="utf-8") as f:
        text = f.read()

    num_tokens, verdict_tokens = _extract_report_tokens(text)

    unprovenanced: list[str] = []

    # 1. Check every extracted number against valid_numbers set
    for tok in num_tokens:
        n_str = tok["value"]
        if not _num_matches_set(n_str, valid_numbers):
            unprovenanced.append(n_str)

    # 2. Check positional binding of each number against recorded number_positions (source, key, value, position)
    expected_num_pos = prov_data.get("number_positions", [])
    if expected_num_pos:
        if len(num_tokens) != len(expected_num_pos):
            # Find which tokens were added/removed/changed
            exp_multiset = [p["value"] for p in expected_num_pos]
            for tok in num_tokens:
                if tok["value"] in exp_multiset:
                    exp_multiset.remove(tok["value"])
                elif tok["value"] not in unprovenanced:
                    unprovenanced.append(tok["value"])
            if not unprovenanced:
                unprovenanced.append(f"number_count_mismatch:{len(num_tokens)}!={len(expected_num_pos)}")
        else:
            for actual_tok, exp_tok in zip(num_tokens, expected_num_pos):
                if actual_tok["value"] != exp_tok["value"] or not exp_tok.get("source") or not exp_tok.get("key_path"):
                    if actual_tok["value"] not in unprovenanced:
                        unprovenanced.append(actual_tok["value"])

    # 3. Check FAIL/PASS verdict strings against recorded verdict_positions
    expected_verdicts = prov_data.get("verdict_positions", [])
    if expected_verdicts:
        if len(verdict_tokens) != len(expected_verdicts):
            unprovenanced.append(f"verdict_count_mismatch:{len(verdict_tokens)}!={len(expected_verdicts)}")
        else:
            for actual_v, exp_v in zip(verdict_tokens, expected_verdicts):
                if actual_v["verdict"] != exp_v["verdict"]:
                    unprovenanced.append(actual_v["verdict"])

    return unprovenanced


CONFIG_KEY_DENYLIST_REGEX = re.compile(
    r"((?:^|_)meas_|verdict|_count_meas|(?:^|_)nees(?:_|$)|(?:^|_)nis(?:_|$)|(?:^|_)v[0-9]_|"
    r"solver_vs_edge|max_hmd_bound_m|false_det_8cam|^deg_|(?:^|_)fa_(?!size_)|(?:^|_)ratio_|"
    r"v0_v01_comparison|hmi_05_faulty_k_sum|provisional_lambda_req)"
)


def find_denylisted_config_keys(data: Any, prefix: str = "") -> list[str]:
    """Recursively scan a loaded config JSON object and return all dotted key paths matching the result-key denylist."""
    violations: list[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            k_str = str(k)
            full_path = f"{prefix}.{k_str}" if prefix else k_str
            if CONFIG_KEY_DENYLIST_REGEX.search(k_str):
                violations.append(full_path)
            if isinstance(v, (dict, list)):
                violations.extend(find_denylisted_config_keys(v, full_path))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            full_path = f"{prefix}.{idx}" if prefix else str(idx)
            if isinstance(item, (dict, list)):
                violations.extend(find_denylisted_config_keys(item, full_path))
    return violations

