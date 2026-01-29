from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests


DEBUG_SEARCH = os.getenv("BIO_WORKFLOW_DEBUG_SEARCH", "").strip().lower() in {"1", "true", "yes", "y"}


def _debug(msg: str) -> None:
    if DEBUG_SEARCH:
        print(msg, flush=True)


def _ncbi_get(url: str, params: Dict[str, str]) -> requests.Response:
    _debug(f"NCBI GET {url} params={params}")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp


def search_pubmed(query: str, retmax: int = 200, email: str = "", tool: str = "bio_workflow") -> List[Dict[str, str]]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    esearch = f"{base}/esearch.fcgi"
    efetch = f"{base}/efetch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(retmax),
        "retmode": "xml",
        "tool": tool,
        "email": email,
    }
    _debug("PubMed esearch start")
    resp = _ncbi_get(esearch, params)
    root = ET.fromstring(resp.text)
    ids = [node.text for node in root.findall(".//Id") if node.text]
    _debug(f"PubMed esearch done: {len(ids)} ids")
    if not ids:
        return []

    results: List[Dict[str, str]] = []
    chunk_size = 200
    for start in range(0, len(ids), chunk_size):
        chunk_ids = ids[start : start + chunk_size]
        _debug(f"PubMed efetch chunk {start // chunk_size + 1} size={len(chunk_ids)}")
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(chunk_ids),
            "retmode": "xml",
            "tool": tool,
            "email": email,
        }
        time.sleep(0.34)
        fetch_resp = _ncbi_get(efetch, fetch_params)
        fetch_root = ET.fromstring(fetch_resp.text)

        for article in fetch_root.findall(".//PubmedArticle"):
            title_node = article.find(".//ArticleTitle")
            title = title_node.text if title_node is not None else ""
            year_node = article.find(".//PubDate/Year")
            year = year_node.text if year_node is not None else ""
            abstract_node = article.find(".//Abstract/AbstractText")
            abstract = abstract_node.text if abstract_node is not None else ""
            pmid_node = article.find(".//PMID")
            pmid = pmid_node.text if pmid_node is not None else ""
            journal_node = article.find(".//Journal/Title")
            journal = journal_node.text if journal_node is not None else ""

            doi = ""
            for aid in article.findall(".//ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = aid.text or ""
                    break

            results.append(
                {
                    "Title": title or "",
                    "Year": year or "",
                    "DOI": doi,
                    "PMID": pmid or "",
                    "Journal": journal or "",
                    "Abstract": abstract or "",
                    "Source": "PubMed",
                }
            )
    return results


def search_europmc(query: str, retmax: int = 200) -> List[Dict[str, str]]:
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    page_size = min(100, retmax)
    page = 1
    results: List[Dict[str, str]] = []

    while len(results) < retmax:
        _debug(f"EuropePMC page {page} (page_size={page_size})")
        params = {
            "query": query,
            "format": "json",
            "pageSize": str(page_size),
            "page": str(page),
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("resultList", {}).get("result", [])
        if not items:
            break
        for item in items:
            title = item.get("title", "") or ""
            year = str(item.get("pubYear", "") or "")
            doi = item.get("doi", "") or ""
            pmid = item.get("pmid", "") or ""
            pmcid = item.get("pmcid", "") or ""
            journal = item.get("journalTitle", "") or ""
            abstract = item.get("abstractText", "") or ""
            source = item.get("source", "") or "EuropePMC"
            src_id = item.get("id", "") or ""
            url_link = ""
            if source and src_id:
                url_link = f"https://europepmc.org/article/{source}/{src_id}"
            pdf_url = ""
            ft_list = item.get("fullTextUrlList", {}).get("fullTextUrl", []) or []
            for ft in ft_list:
                if str(ft.get("documentStyle", "")).lower() == "pdf":
                    pdf_url = ft.get("url", "") or ""
                    break

            results.append(
                {
                    "Title": title,
                    "Year": year,
                    "DOI": doi,
                    "PMID": pmid,
                    "PMCID": pmcid,
                    "Journal": journal,
                    "Abstract": abstract,
                    "Source": "EuropePMC",
                    "URL": url_link,
                    "PDF_URL": pdf_url,
                }
            )
            if len(results) >= retmax:
                break
        if len(items) < page_size:
            break
        page += 1

    return results


def search_scholar_serpapi(query: str, max_results: int = 50) -> List[Dict[str, str]]:
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        _debug("SerpAPI key missing; skipping Scholar search")
        return []

    url = "https://serpapi.com/search"
    params = {
        "engine": "google_scholar",
        "q": query,
        "num": min(max_results, 20),
        "api_key": api_key,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    results: List[Dict[str, str]] = []
    for item in data.get("organic_results", []):
        title = item.get("title", "")
        year = ""
        summary = item.get("snippet", "")
        link = item.get("link", "")
        results.append(
            {
                "Title": title,
                "Year": year,
                "DOI": "",
                "PMID": "",
                "Journal": "",
                "Abstract": summary,
                "Source": "GoogleScholar",
                "URL": link,
            }
        )
    return results


def search_scholar_free(query: str, max_results: int = 50) -> List[Dict[str, str]]:
    try:
        from scholarly import scholarly
    except Exception:
        _debug("scholarly import failed; skipping Scholar search")
        return []

    results: List[Dict[str, str]] = []
    retries = 0
    fetched = 0
    while fetched < max_results and retries <= 2:
        try:
            _debug(f"Scholar free search start (target={max_results})")
            for pub in scholarly.search_pubs(query):
                if fetched >= max_results:
                    break
                bib = pub.get("bib", {}) if isinstance(pub, dict) else {}
                title = bib.get("title", "") or ""
                year = str(bib.get("pub_year", "") or "")
                abstract = bib.get("abstract", "") or bib.get("summary", "") or ""
                doi = bib.get("doi", "") or ""
                link = pub.get("pub_url", "") if isinstance(pub, dict) else ""
                results.append(
                    {
                        "Title": title,
                        "Year": year,
                        "DOI": doi,
                        "PMID": "",
                        "Journal": "",
                        "Abstract": abstract,
                        "Source": "GoogleScholar",
                        "URL": link,
                    }
                )
                fetched += 1
            break
        except Exception:
            retries += 1
            _debug(f"Scholar free search error; retry {retries}")
            time.sleep(2 + retries)
    return results


def search_scholar(query: str, max_results: int = 50, provider: str = "serpapi") -> List[Dict[str, str]]:
    provider = (provider or "").strip().lower()
    if provider == "serpapi":
        return search_scholar_serpapi(query, max_results=max_results)
    if provider == "scholarly":
        return search_scholar_free(query, max_results=max_results)
    return []
