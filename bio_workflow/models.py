from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProtocolRecord:
    Paper_title: str = ""
    Year: str = ""
    DOI: str = ""
    Paper_ID: str = ""
    Virus_or_system: str = ""
    Sample_matrix: str = ""
    Lysis_reagent: str = ""
    Concentration: str = ""
    Incubation_time: str = ""
    Temp_max: Optional[float] = None
    Temp_max_estimated: str = ""
    Additives: str = ""
    Capture_enrichment: str = ""
    Wash_steps: Optional[int] = None
    Elution: str = ""
    Centrifuge: str = ""
    Heat_mode: str = ""
    Heat_purpose: str = ""
    Downstream_reported: str = ""
    Downstream_compatible: Optional[bool] = None
    Downstream_compatibility: List[str] = field(default_factory=list)
    Downstream_compatibility_flag: Optional[bool] = None
    Compat_flag: str = ""
    Wash_free: str = ""
    Centrifugation_free: str = ""
    Heat_free: str = ""
    Methods: str = ""
    Evidence: str = ""
    Provenance_OK: str = ""
    Provenance: str = ""
    Evidence_text: str = ""
    Bucket: str = ""
    Bucket_weight: Optional[float] = None
    Protocol_ID: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "ProtocolRecord":
        return ProtocolRecord(**data)


@dataclass
class PaperSummary:
    Paper_ID: str
    Best_protocol_ID: str
    Best_protocol_score: int
    Best_protocol_Tier: str
    Best_protocol_provenance: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
