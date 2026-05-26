#!/usr/bin/env python3
"""
pdf_keyword_counter.py
----------------------
Simple text-mining baseline: count occurrences of user-specified keywords
(e.g., lysis reagents like 'Triton X-100', 'Tween 20', 'SDS') across a
folder of PDF papers, and rank them by frequency.

This is intended as a *simple baseline* to compare against an AI-assisted
literature-search workflow (i.e., the kind of comparison reviewers often
request to demonstrate that an AI pipeline outperforms naive text mining).

Usage
-----
    # 1) Edit the KEYWORDS list below (or pass --keywords-file), then:
    python pdf_keyword_counter.py /path/to/pdf_folder

    # With options:
    python pdf_keyword_counter.py /path/to/pdf_folder \
        --keywords-file keywords.txt \
        --output results.csv \
        --per-paper per_paper.csv

Dependencies
------------
    pip install pdfplumber pandas
    # (pdfplumber is preferred; the script falls back to pypdf or
    # pdftotext if pdfplumber isn't available.)

Outputs
-------
    - <output>.csv          : keyword -> total mentions, # papers mentioning,
                              avg mentions per mentioning paper
    - <per_paper>.csv       : matrix of keyword counts per paper (long format)
    - Console summary       : ranked table printed to stdout
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# 1) DEFAULT KEYWORDS  --  edit this list, or pass a file with --keywords-file
# ----------------------------------------------------------------------
# Each entry is a "canonical name" plus a list of regex patterns that count
# as a hit for that name. Patterns are matched case-insensitively.
# Use \b for word boundaries; escape special chars in literal strings.
#
# Tip: include common spelling/spacing/punctuation variants so you don't
# under-count (e.g., "Triton X-100" vs "Triton X100" vs "Triton-X 100").
# ----------------------------------------------------------------------
DEFAULT_KEYWORDS: Dict[str, List[str]] = {
    # --- Non-ionic detergents ---
    "Triton X-100":      [r"triton[\s\-]*x[\s\-]*100"],
    "Triton X-114":      [r"triton[\s\-]*x[\s\-]*114"],
    "Tween 20":          [r"tween[\s\-]*20\b", r"polysorbate[\s\-]*20"],
    "Tween 80":          [r"tween[\s\-]*80\b", r"polysorbate[\s\-]*80"],
    "NP-40 / Nonidet P-40": [r"\bnp[\s\-]*40\b", r"nonidet[\s\-]*p[\s\-]*40"],
    "Igepal CA-630":     [r"igepal[\s\-]*ca[\s\-]*630"],
    "Brij-35":           [r"brij[\s\-]*35"],
    "Brij-58":           [r"brij[\s\-]*58"],
    "Digitonin":         [r"\bdigitonin\b"],
    "Saponin":           [r"\bsaponin\b"],
    "Octyl glucoside":   [r"octyl[\s\-]*glucoside", r"\bn?\-?octyl[\s\-]*\u03b2?\-?d?\-?glucopyranoside\b"],

    # --- Ionic detergents ---
    "SDS":               [r"\bsds\b", r"sodium[\s\-]*dodecyl[\s\-]*sulf(?:ate|ate)"],
    "Sodium deoxycholate": [r"sodium[\s\-]*deoxycholate", r"\bdoc\b(?![a-z])"],
    "Sarkosyl":          [r"\bsarkosyl\b", r"sodium[\s\-]*lauroyl[\s\-]*sarcosinate", r"n[\s\-]*lauroylsarcosine"],
    "CTAB":              [r"\bctab\b", r"cetyltrimethylammonium[\s\-]*bromide"],

    # --- Zwitterionic detergents ---
    "CHAPS":             [r"\bchaps\b"],
    "CHAPSO":            [r"\bchapso\b"],
    "Zwittergent":       [r"\bzwittergent\b"],

    # --- Chaotropes / denaturants ---
    "Guanidinium thiocyanate (GITC/GuSCN)": [
        r"guanidin(?:e|ium)[\s\-]*thiocyanate",
        r"\bgu?\s?[sn]cn\b", r"\bgitc\b",
    ],
    "Guanidine hydrochloride (GuHCl)": [
        r"guanidin(?:e|ium)[\s\-]*(?:hydrochloride|hcl)",
        r"\bgu?hcl\b",
    ],
    "Urea":              [r"\burea\b"],
    "Thiourea":          [r"\bthiourea\b"],

    # --- Chelators / reducing agents commonly in lysis buffers ---
    "EDTA":              [r"\bedta\b", r"ethylenediaminetetraacetic"],
    "EGTA":              [r"\begta\b"],
    "DTT":               [r"\bdtt\b", r"dithiothreitol"],
    "TCEP":              [r"\btcep\b", r"tris\(2\-carboxyethyl\)phosphine"],
    "β-mercaptoethanol": [r"\bbme\b", r"beta[\s\-]*mercaptoethanol",
                          r"\u03b2[\s\-]*mercaptoethanol", r"2[\s\-]*mercaptoethanol"],

    # --- Proteases / nucleases used in lysis ---
    "Proteinase K":      [r"proteinase[\s\-]*k"],
    "Lysozyme":          [r"\blysozyme\b"],

    # --- Heat / mechanical lysis (for completeness) ---
    "Heat lysis":        [r"heat[\s\-]*lysis", r"heat[\s\-]*inactivation",
                          r"thermal[\s\-]*lysis"],
    "Freeze-thaw":       [r"freeze[\s\-]*thaw"],

    # --- Commercial lysis-buffer brand names (often appear in viral-diagnostic papers) ---
    "QuickExtract":      [r"quickextract"],
    "RIPA buffer":       [r"\bripa\b[\s\-]*buffer", r"\bripa\b"],

    # --- Generic anchor terms (useful as denominators / sanity checks) ---
    "lysis buffer (generic)": [r"lysis[\s\-]*buffer"],
    "detergent (generic)":    [r"\bdetergent\b"],
}


# ----------------------------------------------------------------------
# 2) PDF -> text
# ----------------------------------------------------------------------
def extract_text_from_pdf(path: Path) -> str:
    """Return all text in the PDF as one big lowercase string.

    Tries pdfplumber, then pypdf, then the `pdftotext` CLI (poppler).
    Returns an empty string if all three fail.
    """
    # --- pdfplumber (best layout handling) ---
    try:
        import pdfplumber
        chunks: List[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                chunks.append(t)
        text = "\n".join(chunks)
        if text.strip():
            return text
    except Exception as e:
        print(f"  [pdfplumber failed on {path.name}: {e}]", file=sys.stderr)

    # --- pypdf fallback ---
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if text.strip():
            return text
    except Exception as e:
        print(f"  [pypdf failed on {path.name}: {e}]", file=sys.stderr)

    # --- pdftotext CLI fallback ---
    try:
        import subprocess
        out = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, timeout=120,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    except Exception as e:
        print(f"  [pdftotext failed on {path.name}: {e}]", file=sys.stderr)

    print(f"  [WARNING] No text extracted from {path.name} "
          f"(scanned PDF? consider OCR)", file=sys.stderr)
    return ""


def normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace + drop soft hyphens that often
    break across-line keyword matches (e.g., 'Tri-\nton X-100')."""
    # Remove hyphen-newline (line-wrapped words like "Tri-\nton")
    text = re.sub(r"-\s*\n\s*", "", text)
    # Collapse all whitespace runs to a single space
    text = re.sub(r"\s+", " ", text)
    return text.lower()


# ----------------------------------------------------------------------
# 3) Counting
# ----------------------------------------------------------------------
def compile_patterns(keywords: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    return {
        name: [re.compile(p, re.IGNORECASE) for p in patterns]
        for name, patterns in keywords.items()
    }


def count_keywords_in_text(
    text: str,
    compiled: Dict[str, List[re.Pattern]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, patterns in compiled.items():
        total = 0
        for pat in patterns:
            total += len(pat.findall(text))
        counts[name] = total
    return counts


# ----------------------------------------------------------------------
# 4) Keyword file loader (optional)
# ----------------------------------------------------------------------
def load_keywords_file(path: Path) -> Dict[str, List[str]]:
    """Load keywords from a text file. Two accepted formats:

    (a) One keyword per line. Each line is treated as both the canonical
        name and a (regex-escaped) literal pattern.

            Triton X-100
            Tween 20
            SDS

    (b) Tab-separated "name<TAB>pattern1<TAB>pattern2..." for full control:

            Triton X-100\ttriton[\\s\\-]*x[\\s\\-]*100
            SDS\t\\bsds\\b\tsodium dodecyl sulfate
    """
    kws: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = line.split("\t")
                name, patterns = parts[0].strip(), [p.strip() for p in parts[1:] if p.strip()]
                kws[name] = patterns or [re.escape(name)]
            else:
                # Treat plain literal; allow flexible whitespace/hyphenation
                literal = re.escape(line)
                # loosen escaped spaces / hyphens so "Triton X-100" matches "Triton X 100"
                literal = literal.replace(r"\ ", r"[\s\-]*").replace(r"\-", r"[\s\-]*")
                kws[line] = [literal]
    return kws


# ----------------------------------------------------------------------
# 5) Main
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Count keyword occurrences across a folder of PDFs."
    )
    ap.add_argument("pdf_folder", type=Path,
                    help="Folder containing PDF files (searched recursively).")
    ap.add_argument("--keywords-file", type=Path, default=None,
                    help="Optional file of keywords (overrides DEFAULT_KEYWORDS).")
    ap.add_argument("--output", type=Path, default=Path("keyword_summary.csv"),
                    help="CSV file for the ranked summary (default: keyword_summary.csv).")
    ap.add_argument("--per-paper", type=Path, default=Path("keyword_per_paper.csv"),
                    help="CSV file for per-paper counts (default: keyword_per_paper.csv).")
    ap.add_argument("--min-hits", type=int, default=0,
                    help="Only print keywords with >= this many total hits in the summary.")
    args = ap.parse_args()

    if not args.pdf_folder.is_dir():
        print(f"ERROR: {args.pdf_folder} is not a directory.", file=sys.stderr)
        return 2

    keywords = (load_keywords_file(args.keywords_file)
                if args.keywords_file else DEFAULT_KEYWORDS)
    compiled = compile_patterns(keywords)

    pdfs = sorted(args.pdf_folder.rglob("*.pdf"))
    if not pdfs:
        print(f"ERROR: No PDF files found under {args.pdf_folder}", file=sys.stderr)
        return 2

    print(f"Found {len(pdfs)} PDF(s) under {args.pdf_folder}")
    print(f"Tracking {len(keywords)} keyword(s).\n")

    # Per-paper counts
    per_paper_rows: List[Dict[str, object]] = []
    totals: Dict[str, int] = defaultdict(int)
    papers_with_hit: Dict[str, int] = defaultdict(int)

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        text = extract_text_from_pdf(pdf)
        if not text.strip():
            continue
        norm = normalize_text(text)
        counts = count_keywords_in_text(norm, compiled)

        for name, c in counts.items():
            totals[name] += c
            if c > 0:
                papers_with_hit[name] += 1

        row = {"paper": pdf.name}
        row.update(counts)
        per_paper_rows.append(row)

    # ------------------ Write per-paper CSV ------------------
    if per_paper_rows:
        fieldnames = ["paper"] + list(keywords.keys())
        with open(args.per_paper, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_paper_rows)
        print(f"\nWrote per-paper counts -> {args.per_paper}")

    # ------------------ Write ranked summary CSV ------------------
    summary_rows: List[Dict[str, object]] = []
    n_papers_with_text = len(per_paper_rows)
    for name in keywords:
        total = totals[name]
        n_papers = papers_with_hit[name]
        summary_rows.append({
            "keyword": name,
            "total_mentions": total,
            "papers_mentioning": n_papers,
            "fraction_of_papers": (n_papers / n_papers_with_text) if n_papers_with_text else 0.0,
            "avg_mentions_per_mentioning_paper": (total / n_papers) if n_papers else 0.0,
        })
    summary_rows.sort(key=lambda r: (-r["total_mentions"], -r["papers_mentioning"]))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote ranked summary  -> {args.output}\n")

    # ------------------ Pretty-print ranked table ------------------
    print(f"=== Ranked keyword counts across {n_papers_with_text} paper(s) ===")
    print(f"{'Rank':>4}  {'Keyword':<40}  {'Total':>7}  {'#Papers':>8}  {'%Papers':>8}")
    print("-" * 76)
    rank = 0
    for row in summary_rows:
        if row["total_mentions"] < args.min_hits:
            continue
        rank += 1
        pct = 100.0 * row["fraction_of_papers"]
        print(f"{rank:>4}  {row['keyword']:<40}  "
              f"{row['total_mentions']:>7}  {row['papers_mentioning']:>8}  {pct:>7.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
