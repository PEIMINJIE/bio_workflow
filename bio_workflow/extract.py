from __future__ import annotations

import base64
import json
import os
import re
from typing import Dict, List

import anthropic

from .utils import chunk_text, stable_id


# Anthropic requires an explicit max_tokens. Protocol extraction returns a JSON
# array that can hold many protocols per paper, so give it generous headroom.
MAX_OUTPUT_TOKENS = 8000


PROMPT_TEMPLATE_STAGE_1_SAFE = """
You are an extraction assistant for published literature. Only summarize non-actionable information.
Do NOT provide step-by-step procedures, amounts, or operational instructions.
Extract high-level reported conditions and flags, and cite provenance.

For each protocol mention, return a JSON object with these fields:
Paper_title,
Wash_free, Centrifugation_free, Heat_free,
Downstream_compatible,
Methods,
Protocol_described, Evidence, Source_page

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

Provenance fields (Protocol_described, Evidence, Source_page):
- Protocol_described: true iff the text actually describes a sample-prep/lysis protocol (methods/procedure); false otherwise (e.g., abstract/intro/results only, or a review).
- Evidence: a SHORT non-actionable supporting quote/phrase from the text that justifies the flags/methods (no amounts, no step-by-step), under ~200 characters. If none, use "".
- Source_page: the page number(s) the evidence comes from, taken from the "Page N." labels or "[PAGE N]" markers in the input (e.g., "7" or "7-8"). If unknown, use "".
- If Protocol_described=false: set Evidence="" and Source_page="".

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

Protocol_described, Evidence, Source_page,

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

D) Provenance (Protocol_described, Evidence, Source_page)
- Protocol_described: true iff the text actually describes a sample-prep/lysis protocol (methods/procedure); false otherwise (e.g., abstract/intro/results only, or a review).
- Evidence: a SHORT non-actionable supporting quote/phrase from the text that justifies the flags/detergents (no amounts, no step-by-step), under ~200 characters. If none, use "".
- Source_page: the page number(s) the evidence comes from, taken from the "Page N." labels or "[PAGE N]" markers in the input (e.g., "7" or "7-8"). If unknown, use "".
- If Protocol_described=false: set Evidence="" and Source_page="".

Return ONLY a JSON array.
"""

PROMPT_TEMPLATE_FULL = """
You are a protocol extraction assistant. Extract every protocol from the provided text and OCR snippets.

For each protocol mention, return a JSON object with these fields:

Paper_title,

Wash_free, Centrifugation_free, Heat_free, Downstream_compatibility,

Detergents_used_for_lysis,        # detergents/surfactants ONLY (names)
Detergent_products_unresolved_for_lysis,

Protocol_described, Evidence, Source_page,

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

D) Provenance (Protocol_described, Evidence, Source_page)
- Protocol_described: true iff the text actually describes a sample-prep/lysis protocol (methods/procedure); false otherwise (e.g., abstract/intro/results only, or a review).
- Evidence: a SHORT non-actionable supporting quote/phrase from the text that justifies the flags/detergents (no amounts, no step-by-step), under ~200 characters. If none, use "".
- Source_page: the page number(s) the evidence comes from, taken from the "Page N." labels or "[PAGE N]" markers in the input (e.g., "7" or "7-8"). If unknown, use "".
- If Protocol_described=false: set Evidence="" and Source_page="".

Return ONLY a JSON array.
"""


def _strip_source_page(prompt: str) -> str:
    """Remove the Source_page field/rule so the model does not emit it."""
    prompt = prompt.replace("Protocol_described, Evidence, Source_page,", "Protocol_described, Evidence,")
    prompt = prompt.replace("Protocol_described, Evidence, Source_page", "Protocol_described, Evidence")
    lines = [ln for ln in prompt.split("\n") if not ln.lstrip().startswith("- Source_page:")]
    prompt = "\n".join(lines)
    prompt = prompt.replace('set Evidence="" and Source_page="".', 'set Evidence="".')
    return prompt


def _select_prompt(stage: str | None, safe_mode: bool, include_source_page: bool = True) -> str:
    normalized = (stage or "").strip().lower()
    if normalized == "stage_1":
        prompt = PROMPT_TEMPLATE_STAGE_1_SAFE if safe_mode else PROMPT_TEMPLATE_STAGE_1_FULL
    else:
        prompt = PROMPT_TEMPLATE_SAFE if safe_mode else PROMPT_TEMPLATE_FULL
    if not include_source_page:
        prompt = _strip_source_page(prompt)
    return prompt


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


def _supports_temperature(model: str) -> bool:
    """Opus 4.8 / 4.7 reject `temperature` (400). Other Claude models accept it."""
    normalized = model.lower()
    return not (normalized.startswith("claude-opus-4-8") or normalized.startswith("claude-opus-4-7"))


def _call_claude(
    client: "anthropic.Anthropic",
    model: str,
    system: str,
    messages: List[Dict[str, object]],
    temperature: float,
    safety_identifier: str | None,
) -> str:
    """Send one request to the Claude Messages API and return concatenated text."""
    req: Dict[str, object] = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": system,
        "messages": messages,
    }
    if safety_identifier:
        req["metadata"] = {"user_id": safety_identifier}
    if _supports_temperature(model):
        req["temperature"] = temperature
    resp = client.messages.create(**req)
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    return text or "[]"


def _provider_for_model(model: str) -> str:
    """Pick the API provider from the model id. gpt*/o1/o3/o4 -> OpenAI, else Anthropic."""
    m = model.lower()
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    return "anthropic"


def _make_client(provider: str):
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openai package not installed; run `pip install openai`") from exc
        return OpenAI(api_key=key)
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def _resolve_safety_identifier(provider: str, safety_identifier: str | None) -> str | None:
    if safety_identifier is not None:
        return safety_identifier
    env = "OPENAI_SAFETY_IDENTIFIER" if provider == "openai" else "ANTHROPIC_SAFETY_IDENTIFIER"
    return os.getenv(env, "").strip() or None


def _call_openai(
    client,
    model: str,
    system: str,
    user_content: object,
    temperature: float,
    safety_identifier: str | None,
) -> str:
    """Send one request to the OpenAI Chat Completions API and return the text."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    req: Dict[str, object] = {"model": model, "messages": messages}
    if safety_identifier:
        req["user"] = safety_identifier
    if not model.lower().startswith("gpt-5"):
        req["temperature"] = temperature
    resp = client.chat.completions.create(**req)
    return resp.choices[0].message.content or "[]"


def _call_llm(
    provider: str,
    client,
    model: str,
    system: str,
    user_content: object,
    temperature: float,
    safety_identifier: str | None,
) -> str:
    """Dispatch a single completion to the right provider. `user_content` is a
    plain string (text) or a list of provider-shaped content blocks (images)."""
    if provider == "openai":
        return _call_openai(client, model, system, user_content, temperature, safety_identifier)
    messages = [{"role": "user", "content": user_content}]
    return _call_claude(client, model, system, messages, temperature, safety_identifier)


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
    include_source_page: bool = True,
) -> List[Dict[str, object]]:
    provider = _provider_for_model(model)
    client = _make_client(provider)
    safety_identifier = _resolve_safety_identifier(provider, safety_identifier)
    chunks = chunk_text(text, max_chars=8000)

    all_protocols: List[Dict[str, object]] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        prompt = _select_prompt(stage, safe_mode, include_source_page)
        system = prompt.strip()
        # Logged messages include the system prompt for parity with prior logs.
        debug_messages = [{"role": "system", "content": system}, {"role": "user", "content": chunk}]
        try:
            content = _call_llm(provider, client, model, system, chunk, temperature, safety_identifier)
        except Exception as exc:
            content = "[]"
            if debug_log_dir:
                _write_debug_log(
                    debug_log_dir,
                    f"text_chunk_{chunk_index:03d}",
                    debug_messages,
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
                    debug_messages,
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
                debug_messages,
                content if not debug_minimize else "",
                parsed=debug_payload,
                minimize=debug_minimize,
            )
        if debug_print:
            print(f"LLM text response (chunk {chunk_index}): {debug_payload}")
        all_protocols.extend(protocols)

    return all_protocols


def _image_block(image_bytes: bytes, provider: str, media_type: str = "image/png") -> Dict[str, object]:
    encoded = base64.standard_b64encode(image_bytes).decode("ascii")
    if provider == "openai":
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}}
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": encoded},
    }


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
    include_source_page: bool = True,
) -> List[Dict[str, object]]:
    if not images:
        return []

    if image_batch_size <= 0:
        image_batch_size = len(images)

    provider = _provider_for_model(model)
    client = _make_client(provider)
    safety_identifier = _resolve_safety_identifier(provider, safety_identifier)

    prompt = _select_prompt(stage, safe_mode, include_source_page)
    system = prompt.strip()
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
            content.append(_image_block(image_bytes, provider))

        # Logged messages include the system prompt for parity with prior logs.
        debug_messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        try:
            content_text = _call_llm(provider, client, model, system, content, temperature, safety_identifier)
        except Exception as exc:
            content_text = "[]"
            if debug_log_path:
                _write_debug_log_file(
                    debug_log_path,
                    debug_messages,
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
                    debug_messages,
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
                debug_messages,
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
            if part.get("type") in ("image", "image_url"):
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
