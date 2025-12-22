"""Vulnerability detection utilities"""

from .dictionary import (
    VulnDictionary,
    VulnerabilityPattern,
    load_vuln_patterns,
    get_default_dictionary,
)
from .rules import (
    VulnRules,
    match_vulnerability,
    extract_vuln_features,
    find_dangerous_calls,
    analyze_pointer_usage,
    score_vulnerability_risk,
    get_vulnerability_summary,
)
from .vuln_lines import extract_vul_line_numbers

__all__ = [
    "VulnDictionary",
    "VulnerabilityPattern",
    "load_vuln_patterns",
    "get_default_dictionary",
    "VulnRules",
    "match_vulnerability",
    "extract_vuln_features",
    "find_dangerous_calls",
    "analyze_pointer_usage",
    "score_vulnerability_risk",
    "get_vulnerability_summary",
    "extract_vul_line_numbers",
]
