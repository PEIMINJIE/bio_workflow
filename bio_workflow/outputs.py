from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def write_protocol_table(records: List[Dict[str, object]], path: Path) -> None:
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)


def write_paper_summary(records: List[Dict[str, object]], path: Path) -> None:
    best_by_paper: Dict[str, Dict[str, object]] = {}
    for r in records:
        paper_id = r.get("Paper_ID", "")
        if not paper_id:
            continue
        if paper_id not in best_by_paper or (r.get("Total_score") or 0) > (best_by_paper[paper_id].get("Total_score") or 0):
            best_by_paper[paper_id] = r

    rows = []
    for paper_id, r in best_by_paper.items():
        rows.append(
            {
                "Paper_ID": paper_id,
                "Best_protocol_ID": r.get("Protocol_ID", ""),
                "Best_protocol_score": r.get("Total_score") or 0,
                "Best_protocol_Tier": r.get("Tier", ""),
                "Best_protocol_provenance": r.get("Provenance", ""),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def write_strict_shortlist(records: List[Dict[str, object]], path: Path, top_n: int = 20) -> None:
    strict = [r for r in records if r.get("Tier") == "Strict"]
    strict.sort(key=lambda x: (-(x.get("Total_score") or 0), x.get("Paper_ID", "")))

    rows = []
    for idx, r in enumerate(strict[:top_n], start=1):
        rows.append(
            {
                "Rank": idx,
                "Paper_ID": r.get("Paper_ID", ""),
                "Protocol_ID": r.get("Protocol_ID", ""),
                "Key_lysis": r.get("Lysis_reagent", ""),
                "Prep_time": r.get("Incubation_time", ""),
                "Downstream_reported": r.get("Downstream_reported", ""),
                "Total_score": r.get("Total_score") or 0,
                "Provenance": r.get("Provenance", ""),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def write_reagent_summary(records: List[Dict[str, object]], path: Path) -> None:
    buckets = defaultdict(list)
    for r in records:
        reagent = str(r.get("Lysis_reagent", "")).strip() or "Unknown"
        buckets[reagent].append(r)

    rows = []
    for reagent, items in buckets.items():
        count_total = len(items)
        count_strict = sum(1 for x in items if x.get("Tier") == "Strict")
        count_heat_required = sum(1 for x in items if str(x.get("Heat_mode", "")).lower() == "required")
        count_isothermal = sum(
            1
            for x in items
            if any(term in str(x.get("Downstream_reported", "")).lower() for term in ["rpa", "lamp", "crispr", "isothermal"])
        )
        dois = []
        for x in items:
            doi = str(x.get("DOI", "")).strip()
            if doi and doi not in dois:
                dois.append(doi)
            if len(dois) >= 3:
                break
        rows.append(
            {
                "Reagent": reagent,
                "Count_total_protocols": count_total,
                "Count_strict_pass": count_strict,
                "Count_heat_required": count_heat_required,
                "Count_isothermal_or_CRISPR_reported": count_isothermal,
                "Representative_refs": "; ".join(dois),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(by=["Count_strict_pass", "Count_total_protocols"], ascending=False, inplace=True)
    df.to_csv(path, index=False)


def _paper_ref(record: Dict[str, object]) -> str:
    title = str(record.get("Paper_title", "")).strip()
    if title:
        return title
    doi = str(record.get("DOI", "")).strip()
    if doi:
        return doi
    paper_id = str(record.get("Paper_ID", "")).strip()
    return paper_id or "Unknown"


def _top_buckets(bucket_counts: Dict[str, int], top_n: int = 3) -> str:
    ordered = sorted(bucket_counts.items(), key=lambda x: (-x[1], x[0]))
    return "; ".join(f"{b} ({c})" for b, c in ordered[:top_n])


def write_method_ranking(records: List[Dict[str, object]], path: Path) -> None:
    method_rows: Dict[str, Dict[str, object]] = {}
    total_weight = 0.0

    for r in records:
        bucket = str(r.get("Bucket", "Unknown"))
        weight = r.get("Bucket_weight")
        if bucket != "Unknown" and weight is not None:
            total_weight += float(weight)

        methods = r.get("Methods", []) or []
        for method in methods:
            if method not in method_rows:
                method_rows[method] = {
                    "raw_count": 0,
                    "weighted_count": 0.0,
                    "bucket_counts": defaultdict(int),
                    "representative": [],
                }
            row = method_rows[method]
            row["raw_count"] += 1
            if bucket != "Unknown" and weight is not None:
                row["weighted_count"] += float(weight)
                row["bucket_counts"][bucket] += 1
            ref = _paper_ref(r)
            if ref not in row["representative"] and len(row["representative"]) < 5:
                row["representative"].append(ref)

    rows = []
    for method, stats in method_rows.items():
        weighted_count = stats["weighted_count"]
        weighted_frequency = (weighted_count / total_weight) if total_weight > 0 else 0.0
        rows.append(
            {
                "method": method,
                "raw_count": stats["raw_count"],
                "weighted_count": round(weighted_count, 4),
                "weighted_frequency": round(weighted_frequency, 4),
                "top_buckets": _top_buckets(stats["bucket_counts"]),
                "representative_papers": "; ".join(stats["representative"]),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(by=["weighted_frequency", "raw_count", "method"], ascending=False, inplace=True)
    df.to_csv(path, index=False)


def _collect_label_stats(records: List[Dict[str, object]], label_field: str) -> List[Dict[str, object]]:
    label_rows: Dict[str, Dict[str, object]] = {}
    total_weight = 0.0

    for r in records:
        bucket = str(r.get("Bucket", "Unknown"))
        weight = r.get("Bucket_weight")
        if bucket != "Unknown" and weight is not None:
            total_weight += float(weight)

        labels = r.get(label_field, []) or []
        for label in labels:
            if label not in label_rows:
                label_rows[label] = {
                    "raw_count": 0,
                    "weighted_count": 0.0,
                    "bucket_counts": defaultdict(int),
                    "representative": [],
                }
            row = label_rows[label]
            row["raw_count"] += 1
            if bucket != "Unknown" and weight is not None:
                row["weighted_count"] += float(weight)
                row["bucket_counts"][bucket] += 1
            ref = _paper_ref(r)
            if ref not in row["representative"] and len(row["representative"]) < 5:
                row["representative"].append(ref)

    rows = []
    for label, stats in label_rows.items():
        weighted_count = stats["weighted_count"]
        weighted_frequency = (weighted_count / total_weight) if total_weight > 0 else 0.0
        rows.append(
            {
                "label": label,
                "raw_count": stats["raw_count"],
                "weighted_count": round(weighted_count, 4),
                "weighted_frequency": round(weighted_frequency, 4),
                "top_buckets": _top_buckets(stats["bucket_counts"]),
                "representative_papers": "; ".join(stats["representative"]),
            }
        )

    rows.sort(key=lambda r: (-(r["weighted_frequency"]), -r["raw_count"], str(r["label"])))
    return rows


def write_label_ranking(
    records: List[Dict[str, object]],
    path: Path,
    label_field: str,
) -> None:
    rows = _collect_label_stats(records, label_field)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def write_unknown_bucket_list(records: List[Dict[str, object]], path: Path) -> None:
    rows = []
    for r in records:
        if str(r.get("Bucket", "")) != "Unknown":
            continue
        rows.append(
            {
                "Paper_ID": r.get("Paper_ID", ""),
                "Protocol_ID": r.get("Protocol_ID", ""),
                "DOI": r.get("DOI", ""),
                "Evidence": r.get("Evidence", ""),
                "Methods": "; ".join(r.get("Methods", []) or []),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
