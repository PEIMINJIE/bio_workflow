# Bio + AI Literature Workflow

This repo implements a complete workflow to search, parse, and score protocols for viral lysis and nucleic acid extraction that are wash-free, heat-free, and centrifugation-free.

## What it does
- Search PubMed (and optionally Google Scholar via SerpAPI) for candidate papers
- Parse PDFs (text + optional OCR for figures/tables)
- Use OpenAI to extract protocol fields + provenance
- Apply strict hard constraints and scoring
- Export four tables:
  - `Protocol_Evidence_Scoring.csv`
  - `Paper_Summary.csv`
  - `Strict_Shortlist_TopN.csv`
  - `Reagent_Summary.csv`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your OpenAI API key (and SerpAPI key if using Google Scholar).

## Configuration

Adjust `config.yaml` for:
- search query
- OCR enable/disable
- temperature threshold
- top N shortlist

## Run the workflow

1) Search PubMed (optional Google Scholar)
```bash
python workflow.py --search
```
This creates:
- `data/cache/search_results_raw.csv`
- `data/cache/candidate_papers.csv`

2) Add PDFs to `data/input/pdfs/`
- Recommended filename format: `Paper_ID__shorttitle.pdf`
- `Paper_ID` matches the ID in `data/cache/candidate_papers.csv`

3) Extract + score protocols
```bash
python workflow.py --extract
```
Outputs are written to `data/output/`.

## Download PDFs automatically (open access)

Europe PMC returns open-access PDF links for some papers. You can download those directly:

```bash
python workflow.py --download-pdfs
```

Config:
- `search.download_max_pdfs`
- `search.download_max_attempts`
- `search.download_timeout_sec`
- `search.download_user_agent`
- `search.download_overwrite`
- `search.download_pmc_fallback` (try PMC PDF when `PMCID` exists)
- `search.download_unpaywall` (use Unpaywall OA PDF links when DOI exists)

Unpaywall requires an email. Set `UNPAYWALL_EMAIL` in `.env`.

Downloads go to `data/input/pdfs/`.

## Output tables

1) Protocol evidence & scoring
- `data/output/Protocol_Evidence_Scoring.csv`

2) Paper-level summary
- `data/output/Paper_Summary.csv`

3) Strict shortlist
- `data/output/Strict_Shortlist_TopN.csv`

4) Reagent summary
- `data/output/Reagent_Summary.csv`

## Notes

- Google Scholar can run in two modes:
  - Free mode via `scholarly`: set `search.use_scholar: true` and `search.scholar_provider: scholarly`.
  - SerpAPI mode: set `SERPAPI_API_KEY` and `search.scholar_provider: serpapi`.
- Europe PMC is free: set `search.use_europmc: true`.
- OCR requires Tesseract installed on your system. If not present, OCR is skipped.
- If Tesseract is missing, the pipeline falls back to text-only extraction.
- If the OpenAI API blocks a prompt for safety, enable `llm.safe_mode: true` (default) to request non-actionable extraction.
- To speed up extraction, set `llm.max_workers` (parallel PDFs). Watch API rate limits.
- The LLM is responsible for evidence extraction and provenance tagging.

## Workflow logic (strict rules)

Hard constraints:
- Wash-free: `Wash_steps == 0`
- Centrifugation-free: `Centrifuge == "No"`
- Heat-free: `Temp_max <= T_threshold` (default 40C)
- Provenance required: `Provenance_OK == "Yes"`

Strict pass:
```
Hard_pass_strict = (Wash_steps==0) AND (Centrifuge=="No") AND
                   (Temp_max<=T_threshold) AND (Provenance_OK=="Yes")
```

Tiering:
- Strict: `Hard_pass_strict == Yes`
- Near-miss: `Hard_pass_strict == No` and `Edit_cost == 1`
- Excluded: `Edit_cost >= 2` or `Provenance_OK == No`

Scoring:
- Start at 100
- -40 for each of: not heat-free, not wash-free, not centrifuge-free
- -20 if only PCR/RT-PCR/qPCR reported (no isothermal/CRISPR)
- -20 if strong inhibitor conditions without dilution/cleanup
- If `Provenance_OK == No`, score = 0

## Troubleshooting
- If the LLM returns invalid JSON, rerun on smaller PDFs or reduce pages in `config.yaml`.
- If provenance is missing, set `Provenance_OK = No` and the protocol will be excluded.
