"""Vulnerability detection rules and feature extraction"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from .dictionary import VulnDictionary, VulnerabilityPattern

logger = logging.getLogger(__name__)

FUNCTION_CALL_PATTERN = re.compile(r'\b(\w+)\s*\(')
POINTER_DECL_PATTERN = re.compile(r'\b(\w+)\s*\*\s*(\w+)')
MALLOC_PATTERN = re.compile(r'\b(malloc|calloc|realloc)\s*\([^)]*\)')
FREE_PATTERN = re.compile(r'\bfree\s*\(\s*(\w+)\s*\)')
NULL_CHECK_PATTERN = re.compile(r'if\s*\([^)]*(?:==|!=)\s*NULL[^)]*\)')
ARRAY_ACCESS_PATTERN = re.compile(r'\b(\w+)\s*\[\s*([^\]]+)\s*\]')
SIZEOF_PATTERN = re.compile(r'\bsizeof\s*\([^)]+\)')


class VulnRules:
    """Rule-based vulnerability detection"""
    
    def __init__(self, dictionary: VulnDictionary):
        self.dictionary = dictionary
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns with error handling"""
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for vuln_type in self.dictionary.vuln_types:
            patterns = self.dictionary.get_patterns(vuln_type)
            compiled = []
            for p in patterns:
                try:
                    compiled.append(re.compile(p))
                except re.error as e:
                    logger.warning(f"Invalid pattern in {vuln_type}: {e}")
            self.compiled_patterns[vuln_type] = compiled
    
    def detect(self, code: str) -> List[Tuple[str, int, str]]:
        """Detect vulnerabilities in code
        
        Returns: List of (vuln_type, line_number, matched_text)
        """
        if not code or not code.strip():
            return []
            
        results = []
        lines = code.split('\n')
        
        for vuln_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for i, line in enumerate(lines):
                    try:
                        match = pattern.search(line)
                        if match:
                            results.append((vuln_type, i + 1, match.group()))
                    except Exception as e:
                        logger.warning(f"Error matching pattern: {e}")
        
        return results
    
    def has_vulnerability(self, code: str) -> bool:
        """Check if code has any vulnerability"""
        return len(self.detect(code)) > 0
    
    def get_vuln_types(self, code: str) -> List[str]:
        """Get list of vulnerability types in code"""
        detections = self.detect(code)
        return list(set(d[0] for d in detections))


def match_vulnerability(code: str, dictionary: VulnDictionary) -> Optional[str]:
    """Quick check for vulnerability type in code"""
    if not code or not code.strip():
        return None
    rules = VulnRules(dictionary)
    types = rules.get_vuln_types(code)
    return types[0] if types else None


def extract_vuln_features(code: str, dictionary: VulnDictionary) -> Dict[str, int]:
    """
    Extract vulnerability features từ code.
    Returns: {'buffer_overflow_count': 2, 'null_pointer_count': 0, ...}
    """
    if not code or not code.strip():
        return {f"{cat}_count": 0 for cat in dictionary.vuln_types}
    
    features: Dict[str, int] = {}
    
    for category in dictionary.vuln_types:
        features[f"{category}_count"] = 0
    
    for pattern in dictionary.patterns:
        matches = pattern.match(code)
        count_key = f"{pattern.category}_count"
        if count_key in features:
            features[count_key] += len(matches)
    
    features["dangerous_function_count"] = 0
    dangerous_funcs = dictionary.get_all_dangerous_functions()
    for match in FUNCTION_CALL_PATTERN.finditer(code):
        func_name = match.group(1)
        if func_name in dangerous_funcs:
            features["dangerous_function_count"] += 1
    
    features["pointer_count"] = len(POINTER_DECL_PATTERN.findall(code))
    features["malloc_count"] = len(MALLOC_PATTERN.findall(code))
    features["free_count"] = len(FREE_PATTERN.findall(code))
    features["null_check_count"] = len(NULL_CHECK_PATTERN.findall(code))
    features["array_access_count"] = len(ARRAY_ACCESS_PATTERN.findall(code))
    
    return features


def find_dangerous_calls(code: str, dictionary: VulnDictionary) -> List[Tuple[str, int, str]]:
    """
    Find all dangerous function calls.
    Returns: [(function_name, line_number, category), ...]
    """
    if not code or not code.strip():
        return []
    
    results: List[Tuple[str, int, str]] = []
    lines = code.split('\n')
    dangerous_funcs = set(dictionary.get_all_dangerous_functions())
    
    for line_num, line in enumerate(lines, 1):
        for match in FUNCTION_CALL_PATTERN.finditer(line):
            func_name = match.group(1)
            if func_name in dangerous_funcs:
                categories = dictionary.get_vuln_type(func_name)
                category = categories[0] if categories else "unknown"
                results.append((func_name, line_num, category))
    
    return results


def analyze_pointer_usage(code: str) -> Dict[str, Any]:
    """
    Analyze pointer declarations, allocations, frees.
    Returns statistics về pointer usage patterns.
    """
    if not code or not code.strip():
        return {
            "pointer_declarations": [],
            "allocations": [],
            "frees": [],
            "null_checks": 0,
            "array_accesses": [],
            "potential_issues": [],
            "stats": {
                "total_pointers": 0,
                "total_allocations": 0,
                "total_frees": 0,
                "alloc_free_ratio": 0.0,
            }
        }
    
    lines = code.split('\n')
    
    pointer_declarations: List[Dict[str, Any]] = []
    for match in POINTER_DECL_PATTERN.finditer(code):
        ptr_type, ptr_name = match.groups()
        line_num = code[:match.start()].count('\n') + 1
        pointer_declarations.append({
            "name": ptr_name,
            "type": ptr_type,
            "line": line_num,
        })
    
    allocations: List[Dict[str, Any]] = []
    for match in MALLOC_PATTERN.finditer(code):
        func_name = match.group(1)
        line_num = code[:match.start()].count('\n') + 1
        allocations.append({
            "function": func_name,
            "line": line_num,
            "code": match.group(0),
        })
    
    frees: List[Dict[str, Any]] = []
    freed_vars: List[str] = []
    for match in FREE_PATTERN.finditer(code):
        var_name = match.group(1)
        line_num = code[:match.start()].count('\n') + 1
        frees.append({
            "variable": var_name,
            "line": line_num,
        })
        freed_vars.append(var_name)
    
    null_checks = len(NULL_CHECK_PATTERN.findall(code))
    
    array_accesses: List[Dict[str, Any]] = []
    for match in ARRAY_ACCESS_PATTERN.finditer(code):
        arr_name, index = match.groups()
        line_num = code[:match.start()].count('\n') + 1
        array_accesses.append({
            "array": arr_name,
            "index": index.strip(),
            "line": line_num,
        })
    
    potential_issues: List[Dict[str, Any]] = []
    
    total_allocs = len(allocations)
    total_frees = len(frees)
    
    if total_allocs > 0 and total_frees == 0:
        potential_issues.append({
            "type": "potential_memory_leak",
            "description": f"Found {total_allocs} allocations but no free() calls",
        })
    
    if total_frees > total_allocs:
        potential_issues.append({
            "type": "potential_double_free",
            "description": f"More frees ({total_frees}) than allocations ({total_allocs})",
        })
    
    for free_info in frees:
        var = free_info["variable"]
        free_line = free_info["line"]
        
        for i, line in enumerate(lines[free_line:], free_line + 1):
            if re.search(rf'\b{re.escape(var)}\s*->', line) or \
               re.search(rf'\*\s*{re.escape(var)}', line):
                potential_issues.append({
                    "type": "potential_use_after_free",
                    "variable": var,
                    "freed_at": free_line,
                    "used_at": i,
                })
                break
    
    for alloc_info in allocations:
        alloc_line = alloc_info["line"]
        next_lines = lines[alloc_line:min(alloc_line + 3, len(lines))]
        has_null_check = any(
            re.search(r'if\s*\([^)]*(?:==|!=)\s*NULL', line) or
            re.search(r'if\s*\(\s*!\s*\w+\s*\)', line)
            for line in next_lines
        )
        if not has_null_check:
            potential_issues.append({
                "type": "missing_null_check",
                "line": alloc_line,
                "allocation": alloc_info["code"],
            })
    
    alloc_free_ratio = total_frees / total_allocs if total_allocs > 0 else 0.0
    
    return {
        "pointer_declarations": pointer_declarations,
        "allocations": allocations,
        "frees": frees,
        "null_checks": null_checks,
        "array_accesses": array_accesses,
        "potential_issues": potential_issues,
        "stats": {
            "total_pointers": len(pointer_declarations),
            "total_allocations": total_allocs,
            "total_frees": total_frees,
            "alloc_free_ratio": round(alloc_free_ratio, 2),
        }
    }


def score_vulnerability_risk(features: Dict[str, int], dictionary: VulnDictionary = None) -> float:
    """
    Calculate overall risk score from features.
    Higher score = more likely vulnerable.
    
    Score range: 0.0 - 1.0
    """
    if not features:
        return 0.0
    
    weights = {
        "buffer_overflow_count": 5.0,
        "use_after_free_count": 5.0,
        "format_string_count": 4.0,
        "null_pointer_count": 3.0,
        "integer_overflow_count": 3.0,
        "dangerous_function_count": 2.0,
        "malloc_count": 0.5,
        "free_count": 0.5,
        "pointer_count": 0.3,
        "array_access_count": 0.2,
    }
    
    mitigations = {
        "null_check_count": -0.5,
    }
    
    raw_score = 0.0
    
    for feature, value in features.items():
        if not isinstance(value, (int, float)):
            continue
        if feature in weights:
            raw_score += weights[feature] * value
        elif feature in mitigations:
            raw_score += mitigations[feature] * value
    
    raw_score = max(0.0, raw_score)
    
    normalized_score = 1.0 - (1.0 / (1.0 + raw_score * 0.1))
    
    return round(min(1.0, max(0.0, normalized_score)), 3)


def get_vulnerability_summary(code: str, dictionary: VulnDictionary) -> Dict[str, Any]:
    """Get comprehensive vulnerability analysis summary"""
    if not code or not code.strip():
        return {
            "has_vulnerabilities": False,
            "vuln_types": [],
            "dangerous_calls": [],
            "features": {},
            "risk_score": 0.0,
            "pointer_analysis": analyze_pointer_usage(""),
        }
    
    rules = VulnRules(dictionary)
    detections = rules.detect(code)
    features = extract_vuln_features(code, dictionary)
    dangerous_calls = find_dangerous_calls(code, dictionary)
    pointer_analysis = analyze_pointer_usage(code)
    risk_score = score_vulnerability_risk(features, dictionary)
    
    return {
        "has_vulnerabilities": len(detections) > 0,
        "vuln_types": list(set(d[0] for d in detections)),
        "detections": [
            {"type": d[0], "line": d[1], "match": d[2]}
            for d in detections
        ],
        "dangerous_calls": [
            {"function": c[0], "line": c[1], "category": c[2]}
            for c in dangerous_calls
        ],
        "features": features,
        "risk_score": risk_score,
        "risk_level": _risk_level(risk_score),
        "pointer_analysis": pointer_analysis,
    }


def _risk_level(score: float) -> str:
    """Convert numeric score to risk level"""
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    elif score > 0.0:
        return "low"
    return "none"
