from __future__ import annotations

import base64
import json
import os
import re
from typing import Dict, List

from openai import OpenAI

from .utils import chunk_text, stable_id


PROMPT_TEMPLATE_STAGE_1_SAFE = """
You are an extraction assistant for published literature. Only summarize non-actionable information.
Do NOT provide step-by-step procedures, amounts, or operational instructions.
Extract high-level reported conditions and flags, and cite provenance.

For each protocol mention, return a JSON object with these fields:
Paper_title,
Wash_free, Centrifugation_free, Heat_free,
Downstream_compatible,
Methods

Rules:
- Wash_free: true iff no wash/rinse/ethanol wash/buffer wash during sample prep.
- Centrifugation_free: true iff no centrifugation during sample prep (quick spin counts as centrifugation).
- Heat_free: true iff sample prep never exceeds 40C.
- Downstream_compatible: true iff the paper explicitly indicates the sample-prep output (e.g., lysate/treated sample)
  can be used directly for downstream amplification/detection (e.g., PCR/qPCR/RPA/LAMP/CRISPR) without reporting
  inhibition and without requiring additional purification/cleanup/inhibitor-removal or other extra processing steps
  to enable downstream reactions.
  - If the paper states downstream inhibition, downstream failure, or the need for purification/cleanup/extraction
    (including kit/column-based purification) before amplification/detection, set Downstream_compatible = false.
  - If downstream compatibility is not explicitly stated or is unclear, set Downstream_compatible = false.
- If no protocol is described, set Wash_free = false, Centrifugation_free = false, Heat_free = false,
  Downstream_compatible = false.

- Methods is a JSON array; choose one or more from:
  detergent_lysis (e.g., Triton X-100 / IGEPAL / NP-40 / Tween; any surfactant-based lysis),
  enzyme_proteinaseK_or_other (Proteinase K or other enzymatic lysis used in sample prep),
  chelating_resin (e.g., Chelex),
  alkaline_pH_shock (NaOH/high pH shock; alkaline lysis),
  heat_lysis_or_inactivation (sample-prep heating >40C used for lysis/inactivation),
  mechanical (bead beating / sonication / freeze-thaw as an explicit sample-prep step),
  other:<free text>

- Prefer the standard labels above. Use other:<free text> ONLY if none of the standard labels apply.
- If a method mentions detergents/surfactants, assign detergent_lysis (do not use other: for that case).
- If a method mentions Proteinase K or enzymatic lysis, assign enzyme_proteinaseK_or_other.
- If a method mentions heat >40C for lysis/inactivation, assign heat_lysis_or_inactivation.
- If a method mentions bead beating/sonication/freeze-thaw, assign mechanical.

Return ONLY a JSON array.
"""

PROMPT_TEMPLATE_STAGE_1_FULL = PROMPT_TEMPLATE_STAGE_1_SAFE

PROMPT_TEMPLATE_SAFE = """
You are an extraction assistant for published literature. Only summarize non-actionable information.
Do NOT provide step-by-step procedures, amounts, or operational instructions.
Extract high-level reported conditions and flags, and cite provenance.

For each protocol mention, return a JSON object with these fields:

Paper_title,

Wash_free, Centrifugation_free, Heat_free, Downstream_compatibility,

Detergents_used_for_lysis,        # detergents/surfactants ONLY (names)
Detergent_products_unresolved_for_lysis,

Rules:

A) 3F flags + Downstream compatibility
- Wash_free: true iff NO wash/rinse/ethanol wash/buffer wash is explicitly mentioned during sample prep/lysis.
- Centrifugation_free: true iff NO centrifugation is explicitly mentioned (quick spin counts).
- Heat_free: true iff sample prep/lysis never exceeds 40C, and this is explicitly stated.
- Downstream_compatibility (array of strings): list only what is explicitly claimed or demonstrated that the resulting lysate/prep is compatible with as a downstream assay input (e.g., "directly used in RT-qPCR", "compatible with RPA", "works with Cas12a detection").
  Allowed examples (not exhaustive): "PCR/qPCR", "RT-PCR/RT-qPCR", "RPA", "LAMP", "CRISPR-Cas12a", "CRISPR-Cas13", "Sequencing".
  If the text claims "compatible with downstream amplification/detection" but does not specify which assay: set Downstream_compatibility=["unspecified_downstream_assay"].
  If not stated: set Downstream_compatibility=[].
- If Protocol_described=false: set Wash_free=false, Centrifugation_free=false, Heat_free=false, Downstream_compatibility=[].

B) Detergents_used_for_lysis (array of strings; detergents/surfactants ONLY)
- Only list detergents/surfactants explicitly used for lysis (e.g., "lysis buffer contained...", "detergent lysis...").
- Do NOT list detergents used only for other purposes (e.g., blocking buffer, assay buffer, wash buffer unrelated to lysis), unless explicitly stated as part of lysis.
- Only list explicitly mentioned detergents/surfactants; do NOT list buffers/salts/chelators/enzymes.
- Normalize common synonyms:
  TX-100 -> Triton X-100
  Triton X100 -> Triton X-100
  Tween20 -> Tween 20
  Tween80 -> Tween 80
  IGEPAL -> IGEPAL CA-630
  NP40 -> NP-40
- Examples of allowed detergent/surfactant names (not exhaustive):
  Triton X-100, Triton X-114, Tween 20, Tween 80, NP-40, IGEPAL CA-630,
  SDS, CHAPS, saponin, digitonin, deoxycholate, N-lauroylsarcosine (Sarkosyl),
  CTAB, Brij-35, Brij-58, Poloxamer 188
- If the text says "detergent-based lysis" but does NOT name the detergent: set Detergents_used_for_lysis=["unspecified_detergent"].
- If Protocol_described=false: set Detergents_used_for_lysis=[].

C) Detergent_products_unresolved_for_lysis (array of strings)
- If a commercial lysis/sample-prep reagent/buffer is mentioned that likely contains detergents but composition is not stated, list it here (e.g., "QuickExtract", "DNA/RNA Shield", "Lucigen extraction solution").
- Do NOT infer its ingredients.
- If the text says "detergent-based lysis" but does NOT name the detergent: set Detergents_used_for_lysis=["unspecified_detergent"].
- If Protocol_described=false: set Detergents_used_for_lysis=[].

Return ONLY a JSON array.
"""

PROMPT_TEMPLATE_FULL = """
You are a protocol extraction assistant. Extract every protocol from the provided text and OCR snippets.

For each protocol mention, return a JSON object with these fields:

Paper_title,

Wash_free, Centrifugation_free, Heat_free, Downstream_compatibility,

Detergents_used_for_lysis,        # detergents/surfactants ONLY (names)
Detergent_products_unresolved_for_lysis,

Rules:

A) 3F flags + Downstream compatibility
- Wash_free: true iff NO wash/rinse/ethanol wash/buffer wash is explicitly mentioned during sample prep/lysis.
- Centrifugation_free: true iff NO centrifugation is explicitly mentioned (quick spin counts).
- Heat_free: true iff sample prep/lysis never exceeds 40C, and this is explicitly stated.
- Downstream_compatibility (array of strings): list only what is explicitly claimed or demonstrated that the resulting lysate/prep is compatible with as a downstream assay input (e.g., "directly used in RT-qPCR", "compatible with RPA", "works with Cas12a detection").
  Allowed examples (not exhaustive): "PCR/qPCR", "RT-PCR/RT-qPCR", "RPA", "LAMP", "CRISPR-Cas12a", "CRISPR-Cas13", "Sequencing".
  If the text claims "compatible with downstream amplification/detection" but does not specify which assay: set Downstream_compatibility=["unspecified_downstream_assay"].
  If not stated: set Downstream_compatibility=[].
- If Protocol_described=false: set Wash_free=false, Centrifugation_free=false, Heat_free=false, Downstream_compatibility=[].

B) Detergents_used_for_lysis (array of strings; detergents/surfactants ONLY)
- Only list detergents/surfactants explicitly used for lysis (e.g., "lysis buffer contained...", "detergent lysis...").
- Do NOT list detergents used only for other purposes (e.g., blocking buffer, assay buffer, wash buffer unrelated to lysis), unless explicitly stated as part of lysis.
- Only list explicitly mentioned detergents/surfactants; do NOT list buffers/salts/chelators/enzymes.
- Normalize common synonyms:
  TX-100 -> Triton X-100
  Triton X100 -> Triton X-100
  Tween20 -> Tween 20
  Tween80 -> Tween 80
  IGEPAL -> IGEPAL CA-630
  NP40 -> NP-40
- Examples of allowed detergent/surfactant names (not exhaustive):
  Triton X-100, Triton X-114, Tween 20, Tween 80, NP-40, IGEPAL CA-630,
  SDS, CHAPS, saponin, digitonin, deoxycholate, N-lauroylsarcosine (Sarkosyl),
  CTAB, Brij-35, Brij-58, Poloxamer 188
- If the text says "detergent-based lysis" but does NOT name the detergent: set Detergents_used_for_lysis=["unspecified_detergent"].
- If Protocol_described=false: set Detergents_used_for_lysis=[].

C) Detergent_products_unresolved_for_lysis (array of strings)
- If a commercial lysis/sample-prep reagent/buffer is mentioned that likely contains detergents but composition is not stated, list it here (e.g., "QuickExtract", "DNA/RNA Shield", "Lucigen extraction solution").
- Do NOT infer its ingredients.
- If the text says "detergent-based lysis" but does NOT name the detergent: set Detergents_used_for_lysis=["unspecified_detergent"].
- If Protocol_described=false: set Detergents_used_for_lysis=[].

Return ONLY a JSON array.
"""


def _select_prompt(stage: str | None, safe_mode: bool) -> str:
    normalized = (stage or "").strip().lower()
    if normalized == "stage_1":
        return PROMPT_TEMPLATE_STAGE_1_SAFE if safe_mode else PROMPT_TEMPLATE_STAGE_1_FULL
    return PROMPT_TEMPLATE_SAFE if safe_mode else PROMPT_TEMPLATE_FULL


def _parse_json_array(text: str) -> List[Dict[str, object]]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Failed to parse JSON array from LLM response")


def extract_protocols_from_text(
    text: str,
    model: str,
    temperature: float,
    safe_mode: bool = True,
    safety_identifier: str | None = None,
    stage: str | None = None,
    debug_log_dir: str | None = None,
    debug_print: bool = False,
    debug_minimize: bool = False,
) -> List[Dict[str, object]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    if safety_identifier is None:
        safety_identifier = os.getenv("OPENAI_SAFETY_IDENTIFIER", "").strip() or None
    chunks = chunk_text(text, max_chars=8000)

    all_protocols: List[Dict[str, object]] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        prompt = _select_prompt(stage, safe_mode)
        messages = [
            {"role": "system", "content": prompt.strip()},
            {"role": "user", "content": chunk},
        ]
        req = {
            "model": model,
            "messages": messages,
        }
        if safety_identifier:
            req["user"] = safety_identifier
        if model.lower().startswith("gpt-5"):
            pass
        else:
            req["temperature"] = temperature
        try:
            resp = client.chat.completions.create(**req)
            content = resp.choices[0].message.content or "[]"
        except Exception as exc:
            content = "[]"
            if debug_log_dir:
                _write_debug_log(
                    debug_log_dir,
                    f"text_chunk_{chunk_index:03d}",
                    messages,
                    content,
                    error=str(exc),
                )
            if debug_print:
                print(f"LLM text extraction failed (chunk {chunk_index}): {exc}")
            continue
        try:
            protocols = _parse_json_array(content)
        except Exception as exc:
            if debug_log_dir:
                _write_debug_log(
                    debug_log_dir,
                    f"text_chunk_{chunk_index:03d}",
                    messages,
                    content,
                    error=str(exc),
                    minimize=debug_minimize,
                )
            if debug_print:
                print(f"LLM text parsing failed (chunk {chunk_index}): {exc}")
            continue
        debug_payload = _minimize_protocols(protocols) if debug_minimize else protocols
        if debug_log_dir:
            _write_debug_log(
                debug_log_dir,
                f"text_chunk_{chunk_index:03d}",
                messages,
                content if not debug_minimize else "",
                parsed=debug_payload,
                minimize=debug_minimize,
            )
        if debug_print:
            print(f"LLM text response (chunk {chunk_index}): {debug_payload}")
        all_protocols.extend(protocols)

    return all_protocols


def _image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_protocols_from_images(
    images: List[bytes],
    model: str,
    temperature: float,
    safe_mode: bool = True,
    safety_identifier: str | None = None,
    stage: str | None = None,
    image_batch_size: int = 6,
    context_text: str | None = None,
    image_labels: List[str] | None = None,
    debug_log_path: str | None = None,
    debug_print: bool = False,
    debug_request_meta: Dict[str, object] | None = None,
    debug_minimize: bool = False,
) -> List[Dict[str, object]]:
    if not images:
        return []

    if image_batch_size <= 0:
        image_batch_size = len(images)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    if safety_identifier is None:
        safety_identifier = os.getenv("OPENAI_SAFETY_IDENTIFIER", "").strip() or None

    prompt = _select_prompt(stage, safe_mode)
    all_protocols: List[Dict[str, object]] = []
    for start in range(0, len(images), image_batch_size):
        batch = images[start : start + image_batch_size]
        content: List[Dict[str, object]] = [
            {"type": "text", "text": "These are sequential pages from a PDF. Extract protocols from the pages and return ONLY a JSON array."}
        ]
        if context_text:
            content.append({"type": "text", "text": f"Paper context (may be incomplete):\n{context_text}"})
        for i, image_bytes in enumerate(batch, start=start):
            content.append({"type": "text", "text": f"Page {i + 1}."})
            content.append({"type": "image_url", "image_url": {"url": _image_bytes_to_data_url(image_bytes)}})

        messages = [
            {"role": "system", "content": prompt.strip()},
            {"role": "user", "content": content},
        ]
        req = {
            "model": model,
            "messages": messages,
        }
        if safety_identifier:
            req["user"] = safety_identifier
        if not model.lower().startswith("gpt-5"):
            req["temperature"] = temperature
        try:
            resp = client.chat.completions.create(**req)
            content_text = resp.choices[0].message.content or "[]"
        except Exception as exc:
            content_text = "[]"
            if debug_log_path:
                _write_debug_log_file(
                    debug_log_path,
                    messages,
                    content_text,
                    error=str(exc),
                    image_labels=image_labels,
                    meta=debug_request_meta,
                )
            if debug_print:
                print(f"LLM image extraction failed: {exc}")
            continue
        try:
            protocols = _parse_json_array(content_text)
        except Exception as exc:
            if debug_log_path:
                _write_debug_log_file(
                    debug_log_path,
                    messages,
                    content_text,
                    error=str(exc),
                    image_labels=image_labels,
                    meta=debug_request_meta,
                    minimize=debug_minimize,
                )
            if debug_print:
                print(f"LLM image parsing failed: {exc}")
            continue
        debug_payload = _minimize_protocols(protocols) if debug_minimize else protocols
        if debug_log_path:
            _write_debug_log_file(
                debug_log_path,
                messages,
                content_text if not debug_minimize else "",
                parsed=debug_payload,
                image_labels=image_labels,
                meta=debug_request_meta,
                minimize=debug_minimize,
            )
        if debug_print:
            print(f"LLM image response: {debug_payload}")
        all_protocols.extend(protocols)

    return all_protocols


def _messages_for_debug(messages: List[Dict[str, object]], image_labels: List[str] | None) -> List[Dict[str, object]]:
    if not image_labels:
        return messages
    cleaned: List[Dict[str, object]] = []
    for m in messages:
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            cleaned.append(m)
            continue
        new_content = []
        image_index = 0
        for part in m["content"]:
            if part.get("type") == "image_url":
                label = image_labels[image_index] if image_index < len(image_labels) else "image"
                new_content.append({"type": "image", "text": f"<image {label}>"})
                image_index += 1
            else:
                new_content.append(part)
        cleaned.append({"role": m.get("role", ""), "content": new_content})
    return cleaned


def _minimize_protocols(protocols: List[Dict[str, object]]) -> List[Dict[str, object]]:
    minimized = []
    for p in protocols:
        minimized.append(
            {
                "Paper_title": p.get("Paper_title", ""),
                "Methods": p.get("Methods", []),
            }
        )
    return minimized


def _write_debug_log(
    debug_dir: str,
    tag: str,
    messages: List[Dict[str, object]],
    response_text: str,
    parsed: List[Dict[str, object]] | None = None,
    error: str | None = None,
    minimize: bool = False,
) -> None:
    payload = {
        "messages": messages,
    }
    if response_text:
        payload["response_text"] = response_text
    if parsed is not None:
        payload["parsed"] = parsed
    if error:
        payload["error"] = error
    if minimize:
        payload["minimized"] = True
    path = os.path.join(debug_dir, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _write_debug_log_file(
    path: str,
    messages: List[Dict[str, object]],
    response_text: str,
    parsed: List[Dict[str, object]] | None = None,
    error: str | None = None,
    image_labels: List[str] | None = None,
    meta: Dict[str, object] | None = None,
    minimize: bool = False,
) -> None:
    payload = {
        "messages": _messages_for_debug(messages, image_labels),
    }
    if response_text:
        payload["response_text"] = response_text
    if parsed is not None:
        payload["parsed"] = parsed
    if error:
        payload["error"] = error
    if meta:
        payload["meta"] = meta
    if minimize:
        payload["minimized"] = True
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def normalize_protocols(protocols: List[Dict[str, object]], paper_meta: Dict[str, str]) -> List[Dict[str, object]]:
    normalized = []
    for p in protocols:
        p = dict(p)
        for key in ["Paper_title", "Year", "DOI", "Paper_ID"]:
            if not p.get(key):
                p[key] = paper_meta.get(key, "")
        if not p.get("Paper_ID"):
            p["Paper_ID"] = stable_id((p.get("Paper_title", "") + p.get("DOI", ""))[:200])
        p["Protocol_ID"] = stable_id(json.dumps(p, sort_keys=True))
        normalized.append(p)
    return normalized
