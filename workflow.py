from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

from bio_workflow.config import load_config
from bio_workflow.download import download_pdfs
from bio_workflow.extract import extract_protocols_from_images, extract_protocols_from_text, normalize_protocols
from bio_workflow.outputs import (
    write_label_ranking,
    write_method_ranking,
    write_unknown_bucket_list,
    write_protocol_table,
)
from bio_workflow.pdf_utils import (
    detect_keyword_pages,
    extract_first_page_snippet,
    extract_pdf_content,
    get_pdf_page_count,
    iter_render_pdf_page_batches_with_plan,
)
from bio_workflow.scoring import score_protocol
from bio_workflow.search import search_europmc, search_pubmed, search_scholar
from bio_workflow.utils import ensure_dir, stable_id

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def _paper_id_from_meta(title: str, year: str, doi: str) -> str:
    if doi:
        doi_short = doi.split("/")[-1][:12].replace(".", "")
        return f"{year}_{doi_short}" if year else doi_short
    return stable_id(f"{title}_{year}")


def _build_review_filter_pattern(terms: List[str]) -> str:
    cleaned = [t.strip() for t in terms if t and t.strip()]
    if not cleaned:
        return ""
    escaped = [re.escape(t) for t in cleaned]
    return r"\b(" + "|".join(escaped) + r")\b"


def _filter_review_papers(df: pd.DataFrame, terms: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    pattern = _build_review_filter_pattern(terms)
    if not pattern:
        return df
    title = df.get("Title", pd.Series([""] * len(df))).fillna("")
    abstract = df.get("Abstract", pd.Series([""] * len(df))).fillna("")
    text = title.astype(str) + " " + abstract.astype(str)
    mask = ~text.str.contains(pattern, case=False, regex=True)
    return df[mask].copy()


def run_search(config_path: Path) -> Path:
    cfg = load_config(config_path)
    ensure_dir(cfg.cache_dir)
    query = cfg.search.get("query", "")
    retmax = int(cfg.search.get("pubmed_retmax", 200))
    use_scholar = bool(cfg.search.get("use_scholar", False))
    scholar_provider = str(cfg.search.get("scholar_provider", "serpapi"))
    use_europmc = bool(cfg.search.get("use_europmc", False))
    europmc_retmax = int(cfg.search.get("europmc_retmax", 200))
    debug_search = os.getenv("BIO_WORKFLOW_DEBUG_SEARCH", "").strip().lower() in {"1", "true", "yes", "y"}
    if debug_search:
        print("Search debug enabled", flush=True)
        print(f"Query length: {len(query)} chars", flush=True)
        print(f"PubMed retmax: {retmax}", flush=True)
        print(f"EuropePMC enabled: {use_europmc} (retmax={europmc_retmax})", flush=True)
        print(f"Scholar enabled: {use_scholar} (provider={scholar_provider})", flush=True)

    if debug_search:
        print("Starting PubMed search...", flush=True)
    pubmed = search_pubmed(query, retmax=retmax, email=os.getenv("NCBI_EMAIL", ""), tool=os.getenv("NCBI_TOOL", "bio_workflow"))
    if debug_search:
        print("PubMed search complete", flush=True)
    if debug_search and use_europmc:
        print("Starting EuropePMC search...", flush=True)
    europmc = search_europmc(query, retmax=europmc_retmax) if use_europmc else []
    if debug_search and use_europmc:
        print("EuropePMC search complete", flush=True)
    if debug_search and use_scholar:
        print("Starting Scholar search...", flush=True)
    scholar = (
        search_scholar(
            query,
            max_results=int(cfg.search.get("scholar_max_results", 50)),
            provider=scholar_provider,
        )
        if use_scholar
        else []
    )
    if debug_search and use_scholar:
        print("Scholar search complete", flush=True)

    print(f"PubMed results: {len(pubmed)}")
    if use_europmc:
        print(f"Europe PMC results: {len(europmc)}")
    if use_scholar:
        print(f"Scholar results ({scholar_provider}): {len(scholar)}")
    combined = pubmed + europmc + scholar
    df = pd.DataFrame(combined)
    review_terms = list(cfg.search.get("review_filter_terms", []))
    review_enabled = bool(cfg.search.get("review_filter_enabled", True))
    before = len(df)
    df = _filter_review_papers(df, review_terms) if review_enabled else df
    removed = before - len(df)
    if review_enabled and removed:
        print(f"Filtered review-type papers: {removed} removed")

    raw_path = cfg.cache_dir / "search_results_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"Wrote raw search results: {raw_path} ({len(df)} rows)")

    df["Paper_ID"] = df.apply(lambda r: _paper_id_from_meta(str(r.get("Title", "")), str(r.get("Year", "")), str(r.get("DOI", ""))), axis=1)
    dedup = df.drop_duplicates(subset=["DOI", "Title"], keep="first")
    candidate_path = cfg.cache_dir / "candidate_papers.csv"
    dedup.to_csv(candidate_path, index=False)
    print(f"Wrote candidate papers: {candidate_path} ({len(dedup)} rows)")
    if len(dedup) == 0:
        print("No results found. Consider loosening the query or using PubMed-only first.")
    return candidate_path


def run_download(config_path: Path, source_csv: Path | None = None) -> None:
    cfg = load_config(config_path)
    config_source = str(cfg.search.get("download_source_csv", "")).strip()
    source_csv = source_csv or (Path(config_source) if config_source else (cfg.cache_dir / "search_results_raw.csv"))
    output_dir = cfg.pdf_input_dir
    max_pdfs = int(cfg.search.get("download_max_pdfs", 200))
    max_attempts = int(cfg.search.get("download_max_attempts", 500))
    timeout_sec = int(cfg.search.get("download_timeout_sec", 60))
    user_agent = str(cfg.search.get("download_user_agent", "bio_workflow/1.0"))
    overwrite = bool(cfg.search.get("download_overwrite", False))
    pmc_fallback = bool(cfg.search.get("download_pmc_fallback", True))
    unpaywall_fallback = bool(cfg.search.get("download_unpaywall", True))
    unpaywall_email = os.getenv("UNPAYWALL_EMAIL", "")
    use_url_field = bool(cfg.search.get("download_use_url_field", False))
    use_doi_resolver = bool(cfg.search.get("download_use_doi_resolver", False))
    proxy_prefix = str(cfg.search.get("download_proxy_prefix", "")).strip()
    ezproxy_browser_auth = bool(cfg.search.get("download_ezproxy_browser_auth", False))
    ezproxy_login_url = str(cfg.search.get("download_ezproxy_login_url", "")).strip()
    ezproxy_headless = bool(cfg.search.get("download_ezproxy_headless", False))
    ezproxy_cookie_file = str(cfg.search.get("download_ezproxy_cookie_file", "")).strip()
    verbose_failures = bool(cfg.search.get("download_verbose_failures", False))
    title_search_enabled = bool(cfg.search.get("download_title_search_enabled", False))
    title_search_max_results = int(cfg.search.get("download_title_search_max_results", 5))
    title_search_engines = list(cfg.search.get("download_title_search_engines", []))
    ezproxy_username = os.getenv("EZPROXY_USERNAME", "")
    ezproxy_password = os.getenv("EZPROXY_PASSWORD", "")
    print(f"Unpaywall email set: {bool(unpaywall_email)}")

    downloaded, attempted = download_pdfs(
        source_csv=source_csv,
        output_dir=output_dir,
        max_pdfs=max_pdfs,
        max_attempts=max_attempts,
        timeout_sec=timeout_sec,
        user_agent=user_agent,
        overwrite=overwrite,
        pmc_fallback=pmc_fallback,
        unpaywall_fallback=unpaywall_fallback,
        unpaywall_email=unpaywall_email,
        use_url_field=use_url_field,
        use_doi_resolver=use_doi_resolver,
        proxy_prefix=proxy_prefix,
        ezproxy_browser_auth=ezproxy_browser_auth,
        ezproxy_login_url=ezproxy_login_url,
        ezproxy_headless=ezproxy_headless,
        ezproxy_username=ezproxy_username,
        ezproxy_password=ezproxy_password,
        ezproxy_cookie_file=ezproxy_cookie_file,
        verbose_failures=verbose_failures,
        title_search_enabled=title_search_enabled,
        title_search_max_results=title_search_max_results,
        title_search_engines=title_search_engines,
    )
    print(f"PDF downloads: {downloaded} / {attempted} attempted")
    if downloaded == 0:
        print("No PDFs downloaded. Ensure PDF_URL exists, PMC is accessible, or Unpaywall email is set.")
    if unpaywall_fallback and not unpaywall_email:
        print("Unpaywall is enabled but UNPAYWALL_EMAIL is not set.")


def _load_candidates(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    mapping: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        paper_id = str(row.get("Paper_ID", ""))
        if paper_id:
            mapping[paper_id] = {
                "Paper_title": str(row.get("Title", "")),
                "Year": str(row.get("Year", "")),
                "DOI": str(row.get("DOI", "")),
                "Paper_ID": paper_id,
            }
    return mapping


def _resolve_paper_meta(pdf_path: Path, candidates: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    stem = pdf_path.stem
    if "__" in stem:
        paper_id = stem.split("__", 1)[0]
        if paper_id in candidates:
            return candidates[paper_id]
    if stem in candidates:
        return candidates[stem]
    return {
        "Paper_title": stem,
        "Year": "",
        "DOI": "",
        "Paper_ID": stable_id(stem),
    }


def run_extract(config_path: Path, candidate_csv: Path | None = None) -> Path:
    cfg = load_config(config_path)
    ensure_dir(cfg.output_dir)
    debug_enabled = bool(cfg.llm.get("debug", False))
    debug_print = bool(cfg.llm.get("debug_print", False))
    debug_minimize = bool(cfg.llm.get("debug_minimize_json", False))
    save_raw_json = bool(cfg.llm.get("save_raw_json", False))
    debug_dir = cfg.output_dir / "debug"
    if debug_enabled:
        ensure_dir(debug_dir)
    raw_json_dir = cfg.output_dir / "llm_json"
    if save_raw_json:
        ensure_dir(raw_json_dir)

    candidate_csv = candidate_csv or (cfg.cache_dir / "candidate_papers.csv")
    candidates = _load_candidates(candidate_csv)

    pdf_dir = cfg.pdf_input_dir
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF input dir not found: {pdf_dir}")

    all_records: List[Dict[str, object]] = []
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return cfg.output_dir / "Protocol_Evidence_Scoring.csv"

    input_mode = str(cfg.pdf.get("input_mode", "text")).strip().lower()
    image_dpi = int(cfg.pdf.get("image_dpi", 200))
    image_batch_size = int(cfg.pdf.get("image_batch_size", 6))
    extract_stage = str(cfg.extract.get("stage", "")).strip().lower()
    method_keywords = list(
        cfg.pdf.get(
            "method_keywords",
            [
                "materials and methods",
                "methods",
                "methodology",
                "procedure",
                "protocol",
                "steps",
                "materials",
            ],
        )
    )

    def _process_pdf(pdf: Path) -> List[Dict[str, object]]:
        paper_meta = _resolve_paper_meta(pdf, candidates)
        max_pages = int(cfg.pdf.get("max_pages", 50))
        model = str(cfg.llm.get("model", "gpt-4o-mini"))
        temperature = float(cfg.llm.get("temperature", 0.0))
        safe_mode = bool(cfg.llm.get("safe_mode", True))
        safety_identifier = str(cfg.llm.get("safety_identifier", "")).strip() or None
        ocr_lang = str(cfg.pdf.get("ocr_language", "eng"))
        debug_save_images = bool(cfg.pdf.get("debug_save_images", False))
        pdf_debug_dir = debug_dir / pdf.stem if debug_enabled else None
        if pdf_debug_dir:
            ensure_dir(pdf_debug_dir)

        def _build_image_batches(total_pages: int, batch_size: int, key_pages: List[int]) -> List[List[int]]:
            if total_pages <= 0:
                return []
            key_set = set(key_pages)
            batches: List[List[int]] = []
            i = 0
            while i < total_pages:
                end = min(i + batch_size, total_pages) - 1
                if end + 1 < total_pages and (end + 1) in key_set and end > i:
                    end -= 1
                if end in key_set and end + 1 < total_pages:
                    end += 1
                batches.append(list(range(i, end + 1)))
                i = end + 1
            return batches

        if input_mode == "image":
            protocols: List[Dict[str, object]] = []
            dpi_candidates = [image_dpi, max(100, image_dpi // 2), 100]
            seen = set()
            dpi_candidates = [dpi for dpi in dpi_candidates if not (dpi in seen or seen.add(dpi))]
            batch_size = image_batch_size
            last_exc: Exception | None = None
            rendered_any = False
            total_pages = min(max_pages, get_pdf_page_count(pdf))
            key_pages = detect_keyword_pages(pdf, max_pages=max_pages, keywords=method_keywords)
            if debug_enabled:
                print(
                    f"PDF {pdf.name}: total_pages={total_pages}, initial_batch_size={batch_size}, "
                    f"dpi_candidates={dpi_candidates}, key_pages={len(key_pages)}"
                )
            context_parts: List[str] = []
            if paper_meta.get("Paper_title"):
                context_parts.append(f"Title: {paper_meta['Paper_title']}")
            if paper_meta.get("Year"):
                context_parts.append(f"Year: {paper_meta['Year']}")
            if paper_meta.get("DOI"):
                context_parts.append(f"DOI: {paper_meta['DOI']}")
            first_page_snippet = extract_first_page_snippet(pdf, lang=ocr_lang)
            if first_page_snippet:
                context_parts.append("First page snippet:\n" + first_page_snippet)
            context_text = "\n".join(context_parts).strip() or None
            for dpi in dpi_candidates:
                while batch_size >= 1:
                    try:
                        batch_protocols: List[Dict[str, object]] = []
                        plan = _build_image_batches(total_pages, batch_size, key_pages)
                        batches = iter_render_pdf_page_batches_with_plan(pdf, plan, dpi=dpi)
                        batch_count = 0
                        for batch_index, (batch_pages, batch_images) in enumerate(batches, start=1):
                            batch_count += 1
                            image_labels: List[str] | None = None
                            debug_log_path = None
                            if pdf_debug_dir:
                                image_labels = []
                                if debug_save_images:
                                    for page_index, image_bytes in zip(batch_pages, batch_images):
                                        image_name = f"batch_{batch_index:03d}_page_{page_index + 1}.png"
                                        image_path = pdf_debug_dir / image_name
                                        image_path.write_bytes(image_bytes)
                                        image_labels.append(str(image_path))
                                debug_log_path = str(
                                    pdf_debug_dir / f"batch_{batch_index:03d}_dpi_{dpi}_size_{len(batch_images)}.json"
                                )
                            batch_protocols.extend(
                                extract_protocols_from_images(
                                    batch_images,
                                    model=model,
                                    temperature=temperature,
                                    safe_mode=safe_mode,
                                    safety_identifier=safety_identifier,
                                    stage=extract_stage or None,
                                    image_batch_size=len(batch_images),
                                    context_text=context_text,
                                    image_labels=image_labels,
                                    debug_log_path=debug_log_path,
                                    debug_print=debug_print,
                                    debug_minimize=debug_minimize,
                                    debug_request_meta={
                                        "pdf": pdf.name,
                                        "pages": [p + 1 for p in batch_pages],
                                        "dpi": dpi,
                                        "batch_size": len(batch_images),
                                        "model": model,
                                    },
                                )
                            )
                        if batch_count > 0:
                            rendered_any = True
                            protocols = batch_protocols
                            last_exc = None
                            break
                        last_exc = RuntimeError("No PDF pages rendered for image extraction")
                    except Exception as exc:
                        last_exc = exc
                    if batch_size <= 1:
                        break
                    batch_size = max(1, batch_size // 2)
                if last_exc is None:
                    break
            if not rendered_any:
                text, ocr_text = extract_pdf_content(
                    pdf,
                    max_pages=max_pages,
                    ocr_enabled=bool(cfg.pdf.get("ocr_enabled", True)),
                    lang=str(cfg.pdf.get("ocr_language", "eng")),
                )
                combined_text = (text + "\n" + ocr_text).strip()
                if not combined_text:
                    return []
                protocols = extract_protocols_from_text(
                    combined_text,
                    model=model,
                    temperature=temperature,
                    safe_mode=safe_mode,
                    safety_identifier=safety_identifier,
                    stage=extract_stage or None,
                    debug_log_dir=str(pdf_debug_dir) if pdf_debug_dir else None,
                    debug_print=debug_print,
                    debug_minimize=debug_minimize,
                )
            elif last_exc is not None:
                raise last_exc
        else:
            text, ocr_text = extract_pdf_content(
                pdf,
                max_pages=max_pages,
                ocr_enabled=bool(cfg.pdf.get("ocr_enabled", True)),
                lang=str(cfg.pdf.get("ocr_language", "eng")),
            )
            combined_text = (text + "\n" + ocr_text).strip()
            if not combined_text:
                return []
            protocols = extract_protocols_from_text(
                combined_text,
                model=model,
                temperature=temperature,
                safe_mode=safe_mode,
                safety_identifier=safety_identifier,
                stage=extract_stage or None,
                debug_log_dir=str(pdf_debug_dir) if pdf_debug_dir else None,
                debug_print=debug_print,
                debug_minimize=debug_minimize,
            )
        if save_raw_json:
            raw_payload = {
                "source_pdf": pdf.name,
                "paper_meta": paper_meta,
                "protocols": protocols,
            }
            raw_path = raw_json_dir / f"{pdf.stem}.json"
            with open(raw_path, "w", encoding="utf-8") as handle:
                json.dump(raw_payload, handle, ensure_ascii=True, indent=2)
        protocols = normalize_protocols(protocols, paper_meta)
        return [score_protocol(p, t_threshold=float(cfg.scoring.get("t_threshold_c", 40))) for p in protocols]

    max_workers = int(cfg.llm.get("max_workers", 1))
    if max_workers <= 1:
        for pdf in tqdm(pdfs, desc="Processing PDFs"):
            all_records.extend(_process_pdf(pdf))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_pdf, pdf): pdf for pdf in pdfs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
                try:
                    all_records.extend(fut.result())
                except Exception as exc:
                    print(f"PDF failed: {futures[fut].name} ({exc})")

    protocol_path = cfg.output_dir / "Protocol_Evidence_Scoring.csv"
    write_protocol_table(all_records, protocol_path)

    label_mode = str(cfg.scoring.get("label_mode", "methods")).strip().lower()
    if extract_stage == "stage_1":
        label_mode = "methods"
    elif extract_stage == "stage_2":
        label_mode = "detergents_capture"
    if label_mode == "detergents_capture":
        detergent_path = cfg.output_dir / "Detergent_Ranking.csv"
        capture_path = cfg.output_dir / "Capture_Ranking.csv"
        write_label_ranking(all_records, detergent_path, "Detergents_used")
        write_label_ranking(all_records, capture_path, "Capture_or_enrichment_or_purification_methods")
    else:
        method_ranking_path = cfg.output_dir / "Method_Ranking.csv"
        write_method_ranking(all_records, method_ranking_path)

    unknown_bucket_path = cfg.output_dir / "Unknown_Bucket_Protocols.csv"
    write_unknown_bucket_list(all_records, unknown_bucket_path)

    return protocol_path


def main() -> None:
    load_dotenv(dotenv_path=Path(".env"), override=False)
    parser = argparse.ArgumentParser(description="Bio + AI literature workflow for wash/heat/centrifuge-free protocols")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--search", action="store_true", help="Run PubMed/Scholar search")
    parser.add_argument("--download-pdfs", action="store_true", help="Download open-access PDFs into data/input/pdfs")
    parser.add_argument("--extract", action="store_true", help="Run PDF parsing + LLM extraction")
    parser.add_argument("--candidates", default="", help="Path to candidate_papers.csv")
    parser.add_argument("--download-source", default="", help="CSV with PDF_URL (default: data/cache/search_results_raw.csv)")

    args = parser.parse_args()
    config_path = Path(args.config)

    if args.search:
        run_search(config_path)

    if args.download_pdfs:
        source_csv = Path(args.download_source) if args.download_source else None
        run_download(config_path, source_csv=source_csv)

    if args.extract:
        candidate_csv = Path(args.candidates) if args.candidates else None
        run_extract(config_path, candidate_csv=candidate_csv)

    if not args.search and not args.extract and not args.download_pdfs:
        parser.print_help()


if __name__ == "__main__":
    main()
