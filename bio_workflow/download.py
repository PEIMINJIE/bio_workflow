from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urljoin
import os
import re
import html

import pandas as pd
import requests
from requests.cookies import RequestsCookieJar

from .utils import ensure_dir, stable_id

DEBUG_DOWNLOAD = os.getenv("BIO_WORKFLOW_DEBUG_DOWNLOAD", "").strip().lower() in {"1", "true", "yes", "y"}


def _debug(msg: str) -> None:
    if DEBUG_DOWNLOAD:
        print(msg, flush=True)


def _normalize_doi(doi: str) -> str:
    doi = (doi or "").strip()
    if not doi:
        return ""
    lowered = doi.lower()
    if "doi.org/" in lowered:
        return doi.split("doi.org/", 1)[1].strip()
    return doi


def _safe_filename(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in ("_", "-"))
    return cleaned[:80] if cleaned else stable_id(name)


def _build_pdf_path(row: Dict[str, str], output_dir: Path) -> Path:
    paper_id = str(row.get("Paper_ID", ""))
    title = str(row.get("Title", ""))
    name = paper_id if paper_id else _safe_filename(title)
    return output_dir / f"{name}.pdf"


def _clean_value(val: object) -> str:
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def _resolve_pmc_pdf_url(pmcid: str, timeout_sec: int, user_agent: str) -> str:
    article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    try:
        resp = requests.get(article_url, timeout=timeout_sec, headers={"User-Agent": user_agent})
        resp.raise_for_status()
    except Exception:
        return ""

    match = re.search(r'href="([^"]+\\.pdf)"', resp.text, re.IGNORECASE)
    if match:
        return urljoin(article_url, match.group(1))
    return ""


def _resolve_unpaywall_pdf_url(doi: str, timeout_sec: int, user_agent: str, email: str) -> str:
    if not doi or not email:
        return ""
    api_url = f"https://api.unpaywall.org/v2/{doi}"
    try:
        resp = requests.get(api_url, params={"email": email}, timeout=timeout_sec, headers={"User-Agent": user_agent})
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return ""

    loc = data.get("best_oa_location") or {}
    pdf_url = loc.get("url_for_pdf") or ""
    if not pdf_url:
        pdf_url = loc.get("url") or ""
    return pdf_url


def _wrap_proxy_url(url: str, proxy_prefix: str) -> str:
    if not url or not proxy_prefix:
        return url
    if url.startswith(proxy_prefix):
        return url
    return f"{proxy_prefix}{url}"


def _extract_pdf_candidates_from_html(html_text: str, base_url: str) -> List[str]:
    candidates: List[str] = []
    normalized = (
        html_text.replace("\\u002F", "/")
        .replace("\\u003A", ":")
        .replace("\\u002D", "-")
        .replace("\\u002E", ".")
    )
    patterns = [
        r'<meta[^>]+(?:name|property)=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
        r'<link[^>]+rel=["\']alternate["\'][^>]+type=["\']application/pdf["\'][^>]+href=["\']([^"\']+)["\']',
        r'["\'](?:pdfUrl|pdf_url|pdfLink)["\']\s*:\s*["\']([^"\']+)["\']',
        r'href=["\']([^"\']+\.pdf[^"\']*)["\']',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, normalized, flags=re.IGNORECASE):
            cleaned = html.unescape(match)
            if cleaned.startswith("javascript:") or cleaned.startswith("mailto:"):
                continue
            candidates.append(urljoin(base_url, cleaned))
    seen: set[str] = set()
    unique: List[str] = []
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def _extract_html_fallback_links(html_text: str, base_url: str) -> List[str]:
    candidates: List[str] = []
    patterns = [
        r'<meta[^>]+name=["\']citation_fulltext_html_url["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']citation_abstract_html_url["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']',
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, html_text, flags=re.IGNORECASE):
            cleaned = html.unescape(match)
            if cleaned.startswith("javascript:") or cleaned.startswith("mailto:"):
                continue
            candidates.append(urljoin(base_url, cleaned))
    seen: set[str] = set()
    unique: List[str] = []
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def _extract_redirect_links(html_text: str, base_url: str) -> List[str]:
    candidates: List[str] = []
    patterns = [
        r'<meta[^>]+http-equiv=["\']refresh["\'][^>]+content=["\'][^;]+;\s*url=([^"\']+)["\']',
        r'window\.location(?:\.href)?\s*=\s*["\']([^"\']+)["\']',
        r'location\.href\s*=\s*["\']([^"\']+)["\']',
        r'url=([^"\'>\s]+)',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, html_text, flags=re.IGNORECASE):
            cleaned = html.unescape(match)
            if cleaned.startswith("javascript:") or cleaned.startswith("mailto:"):
                continue
            candidates.append(urljoin(base_url, cleaned))
    seen: set[str] = set()
    unique: List[str] = []
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def _extract_pdf_links_from_headers(resp: requests.Response) -> List[str]:
    link_header = resp.headers.get("link", "") or resp.headers.get("Link", "")
    if not link_header:
        return []
    candidates: List[str] = []
    for part in link_header.split(","):
        if "application/pdf" not in part.lower():
            continue
        match = re.search(r"<([^>]+)>", part)
        if not match:
            continue
        candidates.append(match.group(1))
    return candidates


def _doi_content_negotiation(
    session: requests.Session, doi: str, timeout_sec: int, proxy_prefix: str
) -> List[str]:
    if not doi:
        return []
    doi_url = f"https://doi.org/{doi}"
    try:
        resp = session.get(doi_url, timeout=timeout_sec, headers={"Accept": "application/pdf"})
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").lower()
        if resp.content and (resp.content.startswith(b"%PDF") or "application/pdf" in content_type):
            return [_wrap_proxy_url(resp.url, proxy_prefix)]
        header_links = _extract_pdf_links_from_headers(resp)
        return [_wrap_proxy_url(url, proxy_prefix) for url in header_links]
    except Exception:
        return []


def _search_pdf_links_semantic_scholar(
    title: str,
    doi: str,
    timeout_sec: int,
    user_agent: str,
    max_results: int,
) -> List[str]:
    if not title and not doi:
        return []
    doi = _normalize_doi(doi)
    query = title
    if doi:
        query = f"\"{title}\" {doi}" if title else doi
    params = {
        "query": query,
        "limit": max(1, min(max_results, 20)),
        "fields": "title,url,openAccessPdf,externalIds",
    }
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            timeout=timeout_sec,
            headers={"User-Agent": user_agent},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    candidates: List[str] = []
    for item in data.get("data", []) or []:
        pdf = item.get("openAccessPdf") or {}
        if pdf.get("url"):
            candidates.append(str(pdf.get("url")))
        if item.get("url"):
            candidates.append(str(item.get("url")))
        ext = item.get("externalIds") or {}
        ext_doi = ext.get("DOI")
        if ext_doi:
            candidates.append(f"https://doi.org/{ext_doi}")
    return candidates


def _search_pdf_links_openalex(
    title: str,
    doi: str,
    timeout_sec: int,
    user_agent: str,
    max_results: int,
) -> List[str]:
    if not title and not doi:
        return []
    doi = _normalize_doi(doi)
    params = {"per_page": max(1, min(max_results, 25))}
    if title:
        params["search"] = title
    elif doi:
        params["filter"] = f"doi:{doi}"
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params=params,
            timeout=timeout_sec,
            headers={"User-Agent": user_agent},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    candidates: List[str] = []
    for item in data.get("results", []) or []:
        open_access = item.get("open_access") or {}
        oa_url = open_access.get("oa_url") or ""
        if oa_url:
            candidates.append(str(oa_url))
        best_oa = item.get("best_oa_location") or {}
        if best_oa.get("pdf_url"):
            candidates.append(str(best_oa.get("pdf_url")))
        if best_oa.get("landing_page_url"):
            candidates.append(str(best_oa.get("landing_page_url")))
        primary = item.get("primary_location") or {}
        if primary.get("pdf_url"):
            candidates.append(str(primary.get("pdf_url")))
        if primary.get("landing_page_url"):
            candidates.append(str(primary.get("landing_page_url")))
        if item.get("doi"):
            candidates.append(str(item.get("doi")))
    return candidates


def _search_pdf_links_crossref(
    title: str,
    doi: str,
    timeout_sec: int,
    user_agent: str,
    max_results: int,
) -> List[str]:
    if not title and not doi:
        return []
    doi = _normalize_doi(doi)
    params = {"rows": max(1, min(max_results, 20))}
    if title:
        params["query.title"] = title
    elif doi:
        params["filter"] = f"doi:{doi}"
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params=params,
            timeout=timeout_sec,
            headers={"User-Agent": user_agent},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    candidates: List[str] = []
    items = (data.get("message") or {}).get("items", []) or []
    for item in items:
        for link in item.get("link", []) or []:
            if str(link.get("content-type", "")).lower() == "application/pdf" and link.get("URL"):
                candidates.append(str(link.get("URL")))
        if item.get("URL"):
            candidates.append(str(item.get("URL")))
        if item.get("DOI"):
            candidates.append(f"https://doi.org/{item.get('DOI')}")
    return candidates


def _search_pdf_links_europmc(
    title: str,
    doi: str,
    timeout_sec: int,
    user_agent: str,
    max_results: int,
) -> List[str]:
    if not title and not doi:
        return []
    doi = _normalize_doi(doi)
    if title and doi:
        query = f'TITLE:"{title}" AND DOI:"{doi}"'
    elif title:
        query = f'TITLE:"{title}"'
    else:
        query = f'DOI:"{doi}"'
    params = {
        "query": query,
        "format": "json",
        "pageSize": max(1, min(max_results, 25)),
        "page": 1,
    }
    try:
        resp = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params=params,
            timeout=timeout_sec,
            headers={"User-Agent": user_agent},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    candidates: List[str] = []
    for item in data.get("resultList", {}).get("result", []) or []:
        ft_list = item.get("fullTextUrlList", {}).get("fullTextUrl", []) or []
        for ft in ft_list:
            if str(ft.get("documentStyle", "")).lower() == "pdf" and ft.get("url"):
                candidates.append(str(ft.get("url")))
        source = item.get("source", "") or "EuropePMC"
        src_id = item.get("id", "") or ""
        if source and src_id:
            candidates.append(f"https://europepmc.org/article/{source}/{src_id}")
    return candidates


def _is_pdf_response(resp: requests.Response) -> bool:
    content_type = (resp.headers.get("content-type") or "").lower()
    return bool(resp.content and (resp.content.startswith(b"%PDF") or "application/pdf" in content_type))


def _fetch(
    session: requests.Session,
    url: str,
    timeout_sec: int,
    referer: str | None = None,
) -> requests.Response:
    headers: Dict[str, str] = {}
    if referer:
        headers["Referer"] = referer
    return session.get(url, timeout=timeout_sec, headers=headers)


def _attempt_pdf_download(
    session: requests.Session,
    url: str,
    timeout_sec: int,
    referer: str | None,
    proxy_prefix: str,
) -> requests.Response | None:
    try:
        resp = _fetch(session, url, timeout_sec, referer=referer)
        resp.raise_for_status()
        if _is_pdf_response(resp):
            return resp
    except Exception:
        return None
    return None

def _cookies_to_jar(cookies: List[Dict[str, object]]) -> RequestsCookieJar:
    jar = RequestsCookieJar()
    for cookie in cookies:
        name = str(cookie.get("name", ""))
        value = str(cookie.get("value", ""))
        if not name:
            continue
        jar.set(
            name,
            value,
            domain=str(cookie.get("domain", "")) or None,
            path=str(cookie.get("path", "/")) or "/",
        )
    return jar


def _load_cookies_from_file(path: Path) -> RequestsCookieJar | None:
    if not path.exists():
        return None
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "cookies" in data:
            data = data["cookies"]
        if not isinstance(data, list):
            return None
        return _cookies_to_jar(data)
    except Exception:
        return None


def _interactive_ezproxy_login(
    login_url: str,
    username: str,
    password: str,
    headless: bool,
) -> RequestsCookieJar | None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("Playwright is not installed; EZproxy login requires `pip install playwright` and `playwright install`.")
        return None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        page.goto(login_url, wait_until="domcontentloaded")

        if username and password:
            filled = False
            candidates = [
                ("#username", "#password"),
                ("input[name='username']", "input[name='password']"),
                ("input[type='email']", "input[type='password']"),
            ]
            for user_sel, pass_sel in candidates:
                if page.query_selector(user_sel) and page.query_selector(pass_sel):
                    page.fill(user_sel, username)
                    page.fill(pass_sel, password)
                    filled = True
                    break
            if filled:
                for submit_sel in ("button[type='submit']", "input[type='submit']"):
                    if page.query_selector(submit_sel):
                        page.click(submit_sel)
                        break

        print("Complete UMN SSO + DUO in the browser, then press Enter here to continue...")
        input()
        cookies = context.cookies()
        browser.close()

    return _cookies_to_jar(cookies)


def download_pdfs(
    source_csv: Path,
    output_dir: Path,
    max_pdfs: int = 200,
    max_attempts: int = 500,
    timeout_sec: int = 60,
    user_agent: str = "bio_workflow/1.0",
    overwrite: bool = False,
    pmc_fallback: bool = True,
    unpaywall_fallback: bool = True,
    unpaywall_email: str = "",
    use_url_field: bool = False,
    use_doi_resolver: bool = False,
    proxy_prefix: str = "",
    ezproxy_browser_auth: bool = False,
    ezproxy_login_url: str = "",
    ezproxy_headless: bool = False,
    ezproxy_username: str = "",
    ezproxy_password: str = "",
    ezproxy_cookie_file: str = "",
    verbose_failures: bool = False,
    title_search_enabled: bool = False,
    title_search_max_results: int = 5,
    title_search_engines: List[str] | None = None,
) -> Tuple[int, int]:
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    ensure_dir(output_dir)
    df = pd.read_csv(source_csv)
    if df.empty:
        return 0, 0

    downloaded = 0
    attempted = 0

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
            "Accept-Language": "en-US,en;q=0.8",
        }
    )
    if ezproxy_cookie_file:
        jar = _load_cookies_from_file(Path(ezproxy_cookie_file))
        if jar:
            session.cookies.update(jar)
    if ezproxy_browser_auth and ezproxy_login_url:
        jar = _interactive_ezproxy_login(
            ezproxy_login_url,
            ezproxy_username,
            ezproxy_password,
            ezproxy_headless,
        )
        if jar:
            session.cookies.update(jar)

    for _, row in df.iterrows():
        if downloaded >= max_pdfs:
            break
        if attempted >= max_attempts:
            break

        candidate_urls: List[str] = []
        pdf_url = _clean_value(row.get("PDF_URL", ""))
        if pdf_url:
            candidate_urls.append(_wrap_proxy_url(pdf_url, proxy_prefix))

        if use_url_field:
            url = _clean_value(row.get("URL", ""))
            if url:
                candidate_urls.append(_wrap_proxy_url(url, proxy_prefix))

        if use_doi_resolver:
            doi = _clean_value(row.get("DOI", ""))
            if doi:
                doi_url = f"https://doi.org/{_normalize_doi(doi)}"
                candidate_urls.append(_wrap_proxy_url(doi_url, proxy_prefix))
                candidate_urls.extend(_doi_content_negotiation(session, doi, timeout_sec, proxy_prefix))

        if title_search_enabled:
            title = _clean_value(row.get("Title", ""))
            doi = _clean_value(row.get("DOI", ""))
            engines = title_search_engines or ["semantic_scholar", "openalex", "crossref", "europmc"]
            for engine in engines:
                engine = (engine or "").strip().lower()
                if engine == "semantic_scholar":
                    search_urls = _search_pdf_links_semantic_scholar(
                        title=title,
                        doi=doi,
                        timeout_sec=timeout_sec,
                        user_agent=user_agent,
                        max_results=title_search_max_results,
                    )
                elif engine == "openalex":
                    search_urls = _search_pdf_links_openalex(
                        title=title,
                        doi=doi,
                        timeout_sec=timeout_sec,
                        user_agent=user_agent,
                        max_results=title_search_max_results,
                    )
                elif engine == "crossref":
                    search_urls = _search_pdf_links_crossref(
                        title=title,
                        doi=doi,
                        timeout_sec=timeout_sec,
                        user_agent=user_agent,
                        max_results=title_search_max_results,
                    )
                elif engine == "europmc":
                    search_urls = _search_pdf_links_europmc(
                        title=title,
                        doi=doi,
                        timeout_sec=timeout_sec,
                        user_agent=user_agent,
                        max_results=title_search_max_results,
                    )
                else:
                    search_urls = []
                for u in search_urls:
                    candidate_urls.append(_wrap_proxy_url(u, proxy_prefix))

        if unpaywall_fallback:
            doi = _clean_value(row.get("DOI", ""))
            if doi:
                unpaywall_url = _resolve_unpaywall_pdf_url(doi, timeout_sec, user_agent, unpaywall_email)
                if unpaywall_url:
                    candidate_urls.append(_wrap_proxy_url(unpaywall_url, proxy_prefix))

        if pmc_fallback:
            pmcid = _clean_value(row.get("PMCID", ""))
            if pmcid:
                pmc_url = _resolve_pmc_pdf_url(pmcid, timeout_sec, user_agent)
                if pmc_url:
                    candidate_urls.append(_wrap_proxy_url(pmc_url, proxy_prefix))
                pmc_base = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                pmc_candidates = [
                    pmc_base + "pdf/",
                    pmc_base + "pdf/" + pmcid + ".pdf",
                    pmc_base + "?pdf=1",
                    pmc_base + "?format=flat",
                ]
                for u in pmc_candidates:
                    candidate_urls.append(_wrap_proxy_url(u, proxy_prefix))

        if not candidate_urls:
            title = _clean_value(row.get("Title", ""))
            doi = _clean_value(row.get("DOI", ""))
            pmcid = _clean_value(row.get("PMCID", ""))
            _debug(f"No URLs for title='{title[:80]}' doi='{doi}' pmcid='{pmcid}'. Trying public fallbacks.")
            if doi and not use_doi_resolver:
                doi_url = f"https://doi.org/{_normalize_doi(doi)}"
                candidate_urls.append(_wrap_proxy_url(doi_url, proxy_prefix))
                candidate_urls.extend(_doi_content_negotiation(session, doi, timeout_sec, proxy_prefix))
            engines = title_search_engines or ["semantic_scholar", "openalex", "crossref", "europmc"]
            for engine in engines:
                engine = (engine or "").strip().lower()
                if engine == "semantic_scholar":
                    search_urls = _search_pdf_links_semantic_scholar(
                        title=title,
                        doi=doi,
                        timeout_sec=timeout_sec,
                        user_agent=user_agent,
                        max_results=title_search_max_results,
                    )
                elif engine == "openalex":
                    search_urls = _search_pdf_links_openalex(
                        title=title,
                        doi=doi,
                        timeout_sec=timeout_sec,
                        user_agent=user_agent,
                        max_results=title_search_max_results,
                    )
                elif engine == "crossref":
                    search_urls = _search_pdf_links_crossref(
                        title=title,
                        doi=doi,
                        timeout_sec=timeout_sec,
                        user_agent=user_agent,
                        max_results=title_search_max_results,
                    )
                elif engine == "europmc":
                    search_urls = _search_pdf_links_europmc(
                        title=title,
                        doi=doi,
                        timeout_sec=timeout_sec,
                        user_agent=user_agent,
                        max_results=title_search_max_results,
                    )
                else:
                    search_urls = []
                for u in search_urls:
                    candidate_urls.append(_wrap_proxy_url(u, proxy_prefix))
            if doi and not unpaywall_fallback:
                unpaywall_url = _resolve_unpaywall_pdf_url(doi, timeout_sec, user_agent, unpaywall_email)
                if unpaywall_url:
                    candidate_urls.append(_wrap_proxy_url(unpaywall_url, proxy_prefix))
            if pmcid and not pmc_fallback:
                pmc_url = _resolve_pmc_pdf_url(pmcid, timeout_sec, user_agent)
                if pmc_url:
                    candidate_urls.append(_wrap_proxy_url(pmc_url, proxy_prefix))

        if not candidate_urls:
            if verbose_failures:
                title = _clean_value(row.get("Title", ""))
                print(f"SKIP (no URLs): {title}")
            continue

        out_path = _build_pdf_path(row, output_dir)
        if out_path.exists() and not overwrite:
            if verbose_failures:
                print(f"SKIP (exists): {out_path.name}")
            continue

        for url in candidate_urls:
            if attempted >= max_attempts or downloaded >= max_pdfs:
                break
            attempted += 1
            try:
                resp = session.get(url, timeout=timeout_sec)
                resp.raise_for_status()
                final_url = resp.url or url
                content_type = (resp.headers.get("content-type") or "").lower()
                if not _is_pdf_response(resp):
                    if "text/html" in content_type:
                        header_pdf_links = _extract_pdf_links_from_headers(resp)
                        if header_pdf_links:
                            fetched = False
                            for discovered in header_pdf_links:
                                pdf_url = _wrap_proxy_url(discovered, proxy_prefix)
                                pdf_resp = _attempt_pdf_download(
                                    session, pdf_url, timeout_sec, referer=final_url, proxy_prefix=proxy_prefix
                                )
                                if pdf_resp:
                                    resp = pdf_resp
                                    fetched = True
                                    break
                            if fetched:
                                with open(out_path, "wb") as f:
                                    f.write(resp.content)
                                downloaded += 1
                                break
                        discovered_list = _extract_pdf_candidates_from_html(resp.text, final_url)
                        if not discovered_list:
                            fallback_links = _extract_html_fallback_links(resp.text, final_url)
                            redirect_links = _extract_redirect_links(resp.text, final_url)
                            fallback_links.extend(redirect_links)
                            if not fallback_links:
                                if verbose_failures:
                                    print(f"NO PDF LINKS: {final_url} ({resp.status_code}, {content_type})")
                                continue
                            fetched = False
                            for link in fallback_links:
                                try:
                                    fallback_resp = session.get(link, timeout=timeout_sec)
                                    fallback_resp.raise_for_status()
                                    fallback_type = (fallback_resp.headers.get("content-type") or "").lower()
                                    if "text/html" not in fallback_type:
                                        continue
                                    pdf_candidates = _extract_pdf_candidates_from_html(
                                        fallback_resp.text, fallback_resp.url or link
                                    )
                                    if not pdf_candidates:
                                        continue
                                    for discovered in pdf_candidates:
                                        pdf_url = _wrap_proxy_url(discovered, proxy_prefix)
                                        pdf_resp = _attempt_pdf_download(
                                            session, pdf_url, timeout_sec, referer=fallback_resp.url or link, proxy_prefix=proxy_prefix
                                        )
                                        if pdf_resp:
                                            resp = pdf_resp
                                            fetched = True
                                            break
                                    if fetched:
                                        break
                                except Exception:
                                    continue
                            if not fetched:
                                if verbose_failures:
                                    print(f"FAILED PDF LINKS: {final_url} ({resp.status_code})")
                                continue
                        fetched = False
                        for discovered in discovered_list:
                            pdf_url = _wrap_proxy_url(discovered, proxy_prefix)
                            pdf_resp = _attempt_pdf_download(
                                session, pdf_url, timeout_sec, referer=final_url, proxy_prefix=proxy_prefix
                            )
                            if pdf_resp:
                                resp = pdf_resp
                                fetched = True
                                break
                        if not fetched:
                            if verbose_failures:
                                print(f"FAILED PDF LINKS: {final_url} ({resp.status_code})")
                            continue
                    else:
                        if verbose_failures:
                            print(f"NON-PDF CONTENT: {final_url} ({resp.status_code}, {content_type})")
                        continue
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                downloaded += 1
                break
            except Exception as exc:
                if verbose_failures:
                    print(f"REQUEST FAILED: {url} ({exc})")
                continue

    return downloaded, attempted
