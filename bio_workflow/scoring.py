from __future__ import annotations

import re
from typing import Dict, Iterable, List


def _to_float(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(val) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _to_bool(val) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if not s or s == "nan":
        return None
    if s in {"yes", "true", "1", "y"}:
        return True
    if s in {"no", "false", "0", "n"}:
        return False
    return None


def _normalize_methods(val: object) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        items = [str(x).strip() for x in val if str(x).strip()]
    else:
        raw = str(val).strip()
        if not raw:
            return []
        sep = ";" if ";" in raw else ","
        items = [x.strip() for x in raw.split(sep) if x.strip()]
    normalized = []
    for item in items:
        lowered = item.lower()
        if lowered.startswith("other:"):
            normalized.append("other:" + item.split(":", 1)[1].strip())
        else:
            normalized.append(lowered)
    return normalized


_DETERGENT_CANONICAL = [
    "Triton X-100",
    "Triton X-114",
    "Tween 20",
    "Tween 80",
    "NP-40",
    "IGEPAL CA-630",
    "SDS",
    "CHAPS",
    "saponin",
    "digitonin",
    "deoxycholate",
    "N-lauroylsarcosine (Sarkosyl)",
    "CTAB",
    "Brij-35",
    "Brij-58",
    "Poloxamer 188",
]
_DETERGENT_KEYS = {
    re.sub(r"[^a-z0-9]+", "", name.lower()): name for name in _DETERGENT_CANONICAL
}


def _split_list(val: object) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    raw = str(val).strip()
    if not raw:
        return []
    sep = ";" if ";" in raw else ","
    return [x.strip() for x in raw.split(sep) if x.strip()]


def _normalize_detergents(val: object) -> List[str]:
    items = _split_list(val)
    normalized: List[str] = []
    seen = set()
    for item in items:
        if item.lower().startswith("other:"):
            cleaned = "other:" + item.split(":", 1)[1].strip()
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
            continue
        key = re.sub(r"[^a-z0-9]+", "", item.lower())
        name = _DETERGENT_KEYS.get(key)
        if name:
            if name not in seen:
                normalized.append(name)
                seen.add(name)
        else:
            cleaned = f"other:{item}"
            if cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
    return normalized


_CAPTURE_CANONICAL = [
    "magnetic_beads_capture_or_purification",
    "silica_column_or_silica_membrane_purification",
    "paper_based_capture",
    "filtration_or_ultrafiltration",
    "membrane_capillary_or_lateralflow_capture",
    "affinity_capture",
    "precipitation_or_flocculation",
    "microfluidic_solid_phase_extraction",
    "electrokinetic_enrichment",
    "density_gradient_or_ultracentrifugation_enrichment",
    "other",
]
_CAPTURE_KEYS = {name.lower(): name for name in _CAPTURE_CANONICAL}


def _normalize_capture_methods(val: object) -> List[str]:
    items = _split_list(val)
    normalized: List[str] = []
    seen = set()
    for item in items:
        if item.lower().startswith("other:"):
            cleaned = "other:" + item.split(":", 1)[1].strip()
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
            continue
        token = re.sub(r"[^a-z0-9]+", "_", item.lower()).strip("_")
        name = _CAPTURE_KEYS.get(token)
        if name:
            if name not in seen:
                normalized.append(name)
                seen.add(name)
        else:
            cleaned = f"other:{item}"
            if cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
    return normalized


def _bucket_weight(bucket: str) -> float:
    if len(bucket) != 4 or any(ch not in "01" for ch in bucket):
        return 0.0
    missing = bucket.count("0")
    return max(0.0, 1.0 - 0.25 * missing)


def score_protocol(record: Dict[str, object], t_threshold: float = 40.0) -> Dict[str, object]:
    temp_max = _to_float(record.get("Temp_max"))
    wash_steps = _to_int(record.get("Wash_steps"))

    wash_free = _to_bool(record.get("Wash_free"))
    if wash_free is None and wash_steps is not None:
        wash_free = wash_steps == 0

    centrifuge = str(record.get("Centrifuge", "")).strip().lower()
    centrifugation_free = _to_bool(record.get("Centrifugation_free"))
    if centrifugation_free is None:
        if centrifuge == "yes":
            centrifugation_free = False
        elif centrifuge == "no":
            centrifugation_free = True

    heat_free = _to_bool(record.get("Heat_free"))
    if heat_free is None and temp_max is not None:
        heat_free = temp_max <= t_threshold

    downstream_compatibility = _to_bool(record.get("Downstream_compatibility"))
    if downstream_compatibility is None and "Downstream_compatible" in record:
        downstream_compatibility = _to_bool(record.get("Downstream_compatible"))
    if downstream_compatibility is None:
        compat_list = record.get("Downstream_compatibility")
        if isinstance(compat_list, list):
            normalized = [str(x).strip() for x in compat_list if str(x).strip()]
            downstream_compatibility = any(
                item != "unspecified_downstream_assay" for item in normalized
            )
        else:
            downstream_compatibility = _to_bool(record.get("Compat_flag"))

    record["Temp_max"] = temp_max
    record["Wash_steps"] = wash_steps
    record["Wash_free"] = wash_free
    record["Centrifugation_free"] = centrifugation_free
    record["Heat_free"] = heat_free

    if (
        wash_free is None
        or centrifugation_free is None
        or heat_free is None
        or downstream_compatibility is None
    ):
        bucket = "Unknown"
    else:
        bucket = (
            f"{int(wash_free)}"
            f"{int(centrifugation_free)}"
            f"{int(heat_free)}"
            f"{int(downstream_compatibility)}"
        )

    record["Bucket"] = bucket
    record["Bucket_weight"] = _bucket_weight(bucket) if bucket != "Unknown" else None
    record["Downstream_compatibility_flag"] = downstream_compatibility
    record["Methods"] = _normalize_methods(record.get("Methods"))
    detergents_raw = record.get("Detergents_used")
    if not detergents_raw:
        detergents_raw = record.get("Detergents_used_for_lysis")
    record["Detergents_used"] = _normalize_detergents(detergents_raw)
    record["Capture_or_enrichment_or_purification_methods"] = _normalize_capture_methods(
        record.get("Capture_or_enrichment_or_purification_methods")
    )

    evidence = record.get("Evidence") or record.get("Evidence_text") or ""
    record["Evidence"] = str(evidence).strip()
    return record


def extract_methods(records: Iterable[Dict[str, object]]) -> List[str]:
    methods = []
    for r in records:
        for m in r.get("Methods", []) or []:
            if m not in methods:
                methods.append(m)
    return methods
