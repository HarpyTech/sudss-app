"""Build patient history context from patients and reports CSVs.

Produces a dict keyed by patient_id with:
- metadata: patient metadata fields from patients.csv
- history: list of report dicts (sorted by date ascending)
- context: a concatenated string (metadata + timeline) suitable for LLM input

Usage (CLI):
  python -m src.generate.patient_history --patients <patients.csv> --reports <reports.csv> --patient-id P100000

Functions are safe with only the Python stdlib (no pandas required).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


def parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # last resort: try to parse year-month-day-like prefix
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def load_patients(patients_csv: str) -> Dict[str, Dict[str, Any]]:
    """Return mapping patient_id -> metadata dict."""
    patients = {}
    p = Path(patients_csv)
    with p.open("r", newline='', encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pid = row.get("patient_id") or row.get("id")
            if not pid:
                continue
            # keep other fields as metadata
            metadata = {k: v for k, v in row.items() if k != "patient_id"}
            patients[pid] = metadata
    return patients


def load_reports(reports_csv: str) -> Dict[str, List[Dict[str, Any]]]:
    """Return mapping patient_id -> list of report dicts (unsorted)."""
    reports_by_patient = defaultdict(list)
    p = Path(reports_csv)
    with p.open("r", newline='', encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pid = row.get("patient_id")
            if not pid:
                continue
            # copy row and attach parsed date for sorting
            r = dict(row)
            r_date = parse_date(r.get("date", ""))
            r["_parsed_date"] = r_date
            reports_by_patient[pid].append(r)
    return reports_by_patient


def build_patient_histories(patients_csv: str, reports_csv: str) -> Dict[str, Dict[str, Any]]:
    """Merge patients and reports into patient histories.

    Result structure:
      { patient_id: { "metadata": {...}, "history": [...], "context": "..." } }
    """
    patients = load_patients(patients_csv)
    reports = load_reports(reports_csv)

    all_patient_ids = set(patients.keys()) | set(reports.keys())
    out = {}
    for pid in sorted(all_patient_ids):
        metadata = patients.get(pid, {})
        patient_reports = reports.get(pid, [])
        # sort by parsed date ascending, keep those without dates at the end
        patient_reports_sorted = sorted(
            patient_reports,
            key=lambda r: (r.get("_parsed_date") is None, r.get("_parsed_date") or datetime.min),
        )

        # build LLM-friendly timeline entries
        timeline_entries = []
        for r in patient_reports_sorted:
            date = r.get("date") or ""
            # choose best textual summary fields available
            pieces = []
            for f in ("summary", "impression", "findings", "text_preview"):
                v = r.get(f)
                if v:
                    pieces.append(v.strip())
            entry_text = "; ".join(pieces) if pieces else ""
            timeline_entries.append({
                "report_id": r.get("report_id"),
                "date": date,
                "image_id": r.get("imaging_id"),
                "text": entry_text,
                # include raw fields for downstream use
                "raw": {k: v for k, v in r.items() if k != "_parsed_date"},
            })

        # construct a context string: metadata then chronological timeline
        meta_lines = [f"{k}: {v}" for k, v in metadata.items()]
        timeline_lines = []
        for t in timeline_entries:
            date = t.get("date") or "unknown date"
            txt = t.get("text") or "(no summary)"
            timeline_lines.append(f"{date} — {txt}")

        context_parts = []
        if meta_lines:
            context_parts.append("Patient metadata:\n" + "\n".join(meta_lines))
        if timeline_lines:
            context_parts.append("Clinical timeline (chronological):\n" + "\n".join(timeline_lines))
        context = "\n\n".join(context_parts)

        out[pid] = {
            "patient_id": pid,
            "metadata": metadata,
            "history": timeline_entries,
            "context": context,
        }

    return out


def get_patient_history(
    patient_histories: Dict[str, Dict[str, Any]], patient_id: str, max_entries: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Fetch a single patient's history; optionally truncates history to last `max_entries` items.

    Returns a dict containing metadata, history, and context (with history truncated if requested).
    """
    p = patient_histories.get(patient_id)
    if not p:
        return None
    if max_entries is None:
        return p
    # copy and truncate history to most recent `max_entries` items
    history = p.get("history", [])
    if not history:
        return p
    truncated = history[-max_entries:]
    # rebuild context
    timeline_lines = [f"{h.get('date','unknown')} — {h.get('text','(no summary)')}" for h in truncated]
    meta_lines = [f"{k}: {v}" for k, v in p.get("metadata", {}).items()]
    context_parts = []
    if meta_lines:
        context_parts.append("Patient metadata:\n" + "\n".join(meta_lines))
    if timeline_lines:
        context_parts.append("Clinical timeline (most recent):\n" + "\n".join(timeline_lines))
    new = {
        "patient_id": p["patient_id"],
        "metadata": p["metadata"],
        "history": truncated,
        "context": "\n\n".join(context_parts),
    }
    return new


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and fetch patient histories from CSVs")
    parser.add_argument("--patients", required=True, help="Path to patients.csv")
    parser.add_argument("--reports", required=True, help="Path to reports.csv")
    parser.add_argument("--patient-id", required=False, help="Patient ID to print")
    parser.add_argument("--max-entries", required=False, type=int, help="Limit history items in context")
    parser.add_argument("--output", required=False, help="Save JSON for the patient to this path")
    parser.add_argument("--export-jsonl", required=False, help="Export full patient histories to this JSONL file (one JSON per line)")
    parser.add_argument("--export-json", required=False, help="Export full patient histories to a single JSON file mapping patient_id -> data")
    parser.add_argument("--export-json-context", required=False, help="Export single JSON mapping patient_id -> context string (Meta Data / History) suitable to feed LLM")
    args = parser.parse_args()

    histories = build_patient_histories(args.patients, args.reports)

    # If user requested a single JSON export (mapping patient_id -> {metadata, history, context})
    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # build mapping where key is patient id and value contains metadata, history, context
        mapping = {
            pid: {"metadata": pdata.get("metadata", {}), "history": pdata.get("history", []), "context": pdata.get("context", "")}
            for pid, pdata in histories.items()
        }
        out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(mapping)} patient histories to {out_path}")

    # If user requested a compact context-string JSON (patient_id -> single string), write that.
    if args.export_json_context:
        out_path = Path(args.export_json_context)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_ctx = {}
        for pid, pdata in histories.items():
            # metadata lines
            meta = pdata.get("metadata", {})
            meta_lines = [f"{k}: {v}" for k, v in meta.items()]
            # consolidate history as plain text: chronological entries joined
            hist = pdata.get("history", [])
            # each history entry: date — text
            hist_lines = []
            for h in hist:
                date = h.get("date") or "unknown date"
                txt = h.get("text") or "(no summary)"
                hist_lines.append(f"{date} — {txt}")

            ctx_str = "Meta Data:\n" + ("\n".join(meta_lines) if meta_lines else "(no metadata)")
            ctx_str += "\n\nHistory:\n" + ("\n\n".join(hist_lines) if hist_lines else "(no history)")
            mapping_ctx[pid] = ctx_str

        out_path.write_text(json.dumps(mapping_ctx, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(mapping_ctx)} patient context strings to {out_path}")

    # If user requested a full JSONL export for all patients, write that next.
    if args.export_jsonl:
        out_path = Path(args.export_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for pid, pdata in histories.items():
                # write the full patient object as one JSON object per line
                fh.write(json.dumps(pdata, ensure_ascii=False) + "\n")
        print(f"Wrote {len(histories)} patient histories to {out_path}")

    # If a specific patient id was requested, print that as before (optionally truncated)
    if args.patient_id:
        p = get_patient_history(histories, args.patient_id, max_entries=args.max_entries)
        if p is None:
            print(f"No patient found with id {args.patient_id}")
        else:
            out_text = json.dumps(p, indent=2, ensure_ascii=False)
            if args.output:
                Path(args.output).write_text(out_text, encoding="utf-8")
                print(f"Wrote patient history to {args.output}")
            else:
                print(out_text)
    else:
        # if no specific patient requested and no export requested, write summary lines to stdout
        if not args.export_jsonl:
            for pid, pdata in histories.items():
                print(json.dumps({"patient_id": pid, "metadata": pdata["metadata"], "history_count": len(pdata["history"])}, ensure_ascii=False))
