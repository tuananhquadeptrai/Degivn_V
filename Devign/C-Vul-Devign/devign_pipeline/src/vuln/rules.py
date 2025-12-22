"""Vulnerability detection rules and feature extraction"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph.cfg import CFG
    from ..graph.dfg import DFG
    from ..parser.tree_sitter_parser import ParseResult
from .dictionary import VulnDictionary, VulnerabilityPattern

logger = logging.getLogger(__name__)

FUNCTION_CALL_PATTERN = re.compile(r'\b(\w+)\s*\(')
POINTER_DECL_PATTERN = re.compile(r'\b(\w+)\s*\*\s*(\w+)')
MALLOC_PATTERN = re.compile(r'\b(malloc|calloc|realloc)\s*\([^)]*\)')
MALLOC_ASSIGN_PATTERN = re.compile(
    r'(\w+)\s*=\s*(?:malloc|calloc|realloc)\s*\(',
    re.MULTILINE
)
FREE_PATTERN = re.compile(r'\bfree\s*\(\s*(\w+)\s*\)')
NULL_CHECK_PATTERN = re.compile(r'if\s*\([^)]*(?:==|!=)\s*NULL[^)]*\)')
ARRAY_ACCESS_PATTERN = re.compile(r'\b(\w+)\s*\[\s*([^\]]+)\s*\]')
SIZEOF_PATTERN = re.compile(r'\bsizeof\s*\([^)]+\)')

NULL_CHECK_VAR_PATTERN = re.compile(
    r'\b(\w+)\s*(?:==|!=)\s*NULL\b|'
    r'\bNULL\s*(?:==|!=)\s*(\w+)\b|'
    r'!\s*(\w+)\s*[)\s]|'
    r'if\s*\(\s*(\w+)\s*\)',
    re.MULTILINE
)

BOUNDS_CHECK_PATTERN = re.compile(
    r'\b(\w+)\s*(?:<|<=|>|>=)\s*(?:\w+|[0-9]+)|'
    r'(?:\w+|[0-9]+)\s*(?:<|<=|>|>=)\s*(\w+)',
    re.MULTILINE
)

CONDITION_LINE_PATTERN = re.compile(
    r'^\s*(?:if|while|for|switch)\s*\(',
    re.MULTILINE
)

# Functions whose return value should be checked
CHECKABLE_FUNCTIONS = {
    'malloc', 'calloc', 'realloc',  # Memory allocation
    'fopen', 'fgets', 'fread', 'fwrite',  # File operations
    'read', 'write', 'recv', 'send',  # I/O
    'socket', 'connect', 'bind', 'listen', 'accept',  # Network
    'pthread_create', 'pthread_mutex_lock',  # Threading
}

# Pattern to find function calls and their context
UNCHECKED_CALL_PATTERN = re.compile(
    r'(?:(\w+)\s*=\s*)?(\w+)\s*\([^)]*\)\s*;',
    re.MULTILINE
)
POINTER_DEREF_PATTERN = re.compile(
    r'\*\s*(\w+)|'           # *ptr
    r'(\w+)\s*->\s*\w+|'     # ptr->member
    r'(\w+)\s*\[\s*[^]]+\]',  # ptr[index] (could be array or pointer)
    re.MULTILINE
)


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


# ============================================
# V2 FEATURES - Pattern-based vulnerability detection
# ============================================


def _build_graph_context(
    code: str,
    parse_result: Optional['ParseResult'] = None,
    cfg: Optional['CFG'] = None,
    dfg: Optional['DFG'] = None,
) -> Tuple[Optional['ParseResult'], Optional['CFG'], Optional['DFG']]:
    """Build or reuse parse_result, cfg, dfg. Returns (parse_result, cfg, dfg)."""
    if parse_result is None:
        try:
            from ..ast.parser import CFamilyParser
            parser = CFamilyParser()
            parse_result = parser.parse_with_fallback(code)
        except Exception:
            parse_result = None
    
    if parse_result and parse_result.nodes:
        if cfg is None:
            try:
                from ..graphs.cfg import CFGBuilder
                cfg = CFGBuilder().build(parse_result)
            except Exception:
                cfg = None
        
        if dfg is None:
            try:
                from ..graphs.dfg import DFGBuilder
                dfg = DFGBuilder().build(parse_result)
            except Exception:
                dfg = None
    
    return parse_result, cfg, dfg


def _compute_basic_size_metrics(code: str) -> Dict[str, float]:
    """Compute loc and stmt_count."""
    lines = code.split('\n')
    loc = sum(1 for line in lines if line.strip() and not line.strip().startswith('//'))
    stmt_count = code.count(';') + code.count('{')
    return {
        'loc': float(loc),
        'stmt_count': float(stmt_count),
    }


def _index_conditions_and_checks(
    code: str,
    parse_result: Optional['ParseResult'],
) -> Tuple[Set[str], Set[str], Set[int]]:
    """
    Index null checks and bounds checks in the code.
    
    Returns:
        null_checked_vars: Set of variable names that have been null-checked
        bounds_checked_vars: Set of variable names that have been bounds-checked
        condition_lines: Set of line numbers that contain conditions
    """
    null_checked_vars: Set[str] = set()
    bounds_checked_vars: Set[str] = set()
    condition_lines: Set[int] = set()
    
    if not code:
        return null_checked_vars, bounds_checked_vars, condition_lines
    
    lines = code.split('\n')
    
    # Try AST-based detection first
    if parse_result and parse_result.nodes:
        for node in parse_result.nodes:
            # ASTNode is a dataclass with .node_type, .start_line, .text attributes
            node_type = getattr(node, 'node_type', '')
            
            # Find condition lines
            if node_type in ('if_statement', 'while_statement', 'for_statement', 
                            'conditional_expression', 'switch_statement'):
                line = getattr(node, 'start_line', 0)
                if line > 0:
                    condition_lines.add(line)
            
            # Find binary expressions that might be checks
            if node_type == 'binary_expression':
                text = getattr(node, 'text', '') or ''
                
                # Null checks
                null_matches = NULL_CHECK_VAR_PATTERN.findall(text)
                for match in null_matches:
                    for var in match:
                        if var and var.isidentifier():
                            null_checked_vars.add(var)
                
                # Bounds checks
                bounds_matches = BOUNDS_CHECK_PATTERN.findall(text)
                for match in bounds_matches:
                    for var in match:
                        if var and var.isidentifier() and not var.isdigit():
                            bounds_checked_vars.add(var)
    
    # Fallback/supplement with regex on raw code
    for i, line in enumerate(lines, 1):
        # Condition lines
        if CONDITION_LINE_PATTERN.search(line):
            condition_lines.add(i)
        
        # Null checks
        null_matches = NULL_CHECK_VAR_PATTERN.findall(line)
        for match in null_matches:
            for var in match:
                if var and var.isidentifier():
                    null_checked_vars.add(var)
        
        # Bounds checks (only in condition context)
        if 'if' in line or 'while' in line or 'for' in line:
            bounds_matches = BOUNDS_CHECK_PATTERN.findall(line)
            for match in bounds_matches:
                for var in match:
                    if var and var.isidentifier() and not var.isdigit():
                        bounds_checked_vars.add(var)
    
    # Filter out common false positives
    false_positives = {'if', 'while', 'for', 'return', 'int', 'char', 'void', 
                       'sizeof', 'NULL', 'true', 'false'}
    null_checked_vars -= false_positives
    bounds_checked_vars -= false_positives
    
    return null_checked_vars, bounds_checked_vars, condition_lines


def _compute_dangerous_call_features(
    code: str,
    dictionary: VulnDictionary,
    parse_result: Optional['ParseResult'],
    cfg: Optional['CFG'],
    dfg: Optional['DFG'],
    null_checked_vars: Set[str],
    bounds_checked_vars: Set[str],
) -> Dict[str, float]:
    """
    Compute dangerous call features.
    A call is "without check" if its arguments are not validated before the call.
    """
    dangerous_funcs = dictionary.get_all_dangerous_functions()
    
    # Find all dangerous calls using regex
    call_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(f) for f in dangerous_funcs) + r')\s*\([^)]*\)',
        re.MULTILINE
    )
    
    total_calls = 0
    unchecked_calls = 0
    
    for match in call_pattern.finditer(code):
        total_calls += 1
        func_name = match.group(1)
        call_text = match.group(0)
        
        # Extract arguments (simple parsing)
        args_start = call_text.find('(') + 1
        args_end = call_text.rfind(')')
        if args_start > 0 and args_end > args_start:
            args_text = call_text[args_start:args_end]
            # Extract variable names from arguments
            arg_vars = set(re.findall(r'\b([a-zA-Z_]\w*)\b', args_text))
            arg_vars -= {'sizeof', 'NULL', 'true', 'false'}
            
            # Check if any argument variable was null/bounds checked
            has_null_check = bool(arg_vars & null_checked_vars)
            has_bounds_check = bool(arg_vars & bounds_checked_vars)
            
            if not has_null_check and not has_bounds_check:
                unchecked_calls += 1
        else:
            unchecked_calls += 1
    
    ratio = unchecked_calls / max(total_calls, 1)
    
    return {
        'dangerous_call_count': float(total_calls),
        'dangerous_call_without_check_count': float(unchecked_calls),
        'dangerous_call_without_check_ratio': ratio,
    }


def _compute_pointer_deref_features(
    code: str,
    dfg: Optional['DFG'],
    cfg: Optional['CFG'],
    null_checked_vars: Set[str],
) -> Dict[str, float]:
    """
    Compute pointer dereference features.
    A deref is "without null check" if the pointer wasn't checked before use.
    """
    total_derefs = 0
    unchecked_derefs = 0
    
    # Use DFG if available
    if dfg and hasattr(dfg, 'nodes'):
        for node in dfg.nodes:
            if hasattr(node, 'access_type'):
                access_type = str(node.access_type)
                if 'DEREF' in access_type or 'deref' in access_type:
                    total_derefs += 1
                    var_name = getattr(node, 'var_name', '')
                    if var_name not in null_checked_vars:
                        unchecked_derefs += 1
    
    # Fallback/supplement with regex
    if total_derefs == 0:
        for match in POINTER_DEREF_PATTERN.finditer(code):
            total_derefs += 1
            # Get the pointer variable name
            ptr_var = match.group(1) or match.group(2) or match.group(3)
            if ptr_var and ptr_var not in null_checked_vars:
                unchecked_derefs += 1
    
    ratio = unchecked_derefs / max(total_derefs, 1)
    
    return {
        'pointer_deref_count': float(total_derefs),
        'pointer_deref_without_null_check_count': float(unchecked_derefs),
        'pointer_deref_without_null_check_ratio': ratio,
    }


def _compute_array_access_features(
    code: str,
    parse_result: Optional['ParseResult'],
    cfg: Optional['CFG'],
    bounds_checked_vars: Set[str],
) -> Dict[str, float]:
    """
    Compute array access features.
    An access is "without bounds check" if the index variable wasn't validated.
    """
    total_accesses = 0
    unchecked_accesses = 0
    
    for match in ARRAY_ACCESS_PATTERN.finditer(code):
        array_name = match.group(1)
        index_expr = match.group(2)
        
        if array_name in {'sizeof', 'typeof', 'if', 'while', 'for', 'switch'}:
            continue
        
        total_accesses += 1
        
        index_vars = set(re.findall(r'\b([a-zA-Z_]\w*)\b', index_expr))
        index_vars -= {'sizeof', 'NULL', 'true', 'false'}
        index_vars = {v for v in index_vars if not v.isdigit()}
        
        if not index_vars:
            continue
        
        has_bounds_check = bool(index_vars & bounds_checked_vars)
        if not has_bounds_check:
            unchecked_accesses += 1
    
    ratio = unchecked_accesses / max(total_accesses, 1)
    
    return {
        'array_access_count': float(total_accesses),
        'array_access_without_bounds_check_count': float(unchecked_accesses),
        'array_access_without_bounds_check_ratio': ratio,
    }


def _compute_malloc_free_features(
    code: str,
    parse_result: Optional['ParseResult'],
    dfg: Optional['DFG'],
    null_checked_vars: Set[str],
) -> Dict[str, float]:
    """
    Compute malloc/free pattern features.
    - malloc_without_free: allocation without corresponding free (potential leak)
    - free_without_null_check: free() without checking if pointer is null
    """
    allocated_vars: Set[str] = set()
    for match in MALLOC_ASSIGN_PATTERN.finditer(code):
        var_name = match.group(1)
        if var_name and var_name.isidentifier():
            allocated_vars.add(var_name)
    
    freed_vars: Set[str] = set()
    free_count = 0
    free_without_null_check_count = 0
    
    for match in FREE_PATTERN.finditer(code):
        var_name = match.group(1)
        if var_name and var_name.isidentifier():
            freed_vars.add(var_name)
            free_count += 1
            
            if var_name not in null_checked_vars:
                free_without_null_check_count += 1
    
    malloc_count = len(allocated_vars)
    malloc_without_free = len(allocated_vars - freed_vars)
    
    return {
        'malloc_count': float(malloc_count),
        'malloc_without_free_count': float(malloc_without_free),
        'malloc_without_free_ratio': malloc_without_free / max(malloc_count, 1),
        'free_count': float(free_count),
        'free_without_null_check_count': float(free_without_null_check_count),
        'free_without_null_check_ratio': free_without_null_check_count / max(free_count, 1),
    }


def _compute_unchecked_return_features(
    code: str,
    parse_result: Optional['ParseResult'],
    dictionary: VulnDictionary,
) -> Dict[str, float]:
    """
    Compute unchecked return value features.
    A call is "unchecked" if its return value is neither:
    - Assigned to a variable
    - Used in a condition
    """
    checkable_calls = 0
    unchecked_calls = 0
    
    lines = code.split('\n')
    
    for i, line in enumerate(lines):
        for match in UNCHECKED_CALL_PATTERN.finditer(line):
            assigned_var = match.group(1)
            func_name = match.group(2)
            
            if func_name not in CHECKABLE_FUNCTIONS:
                continue
            
            checkable_calls += 1
            
            if assigned_var:
                var_checked = False
                for j in range(i + 1, min(i + 11, len(lines))):
                    check_line = lines[j]
                    if re.search(rf'\b{re.escape(assigned_var)}\b', check_line):
                        if any(kw in check_line for kw in ['if', 'while', '==', '!=', '<', '>', '!']):
                            var_checked = True
                            break
                
                if not var_checked:
                    unchecked_calls += 1
            else:
                unchecked_calls += 1
    
    ratio = unchecked_calls / max(checkable_calls, 1)
    
    return {
        'unchecked_return_value_count': float(unchecked_calls),
        'unchecked_return_value_ratio': ratio,
    }


def extract_vuln_features_v2(
    code: str,
    dictionary: VulnDictionary,
    parse_result: Optional['ParseResult'] = None,
    cfg: Optional['CFG'] = None,
    dfg: Optional['DFG'] = None,
) -> Dict[str, float]:
    """
    Graph/AST-based vulnerability features focused on missing defenses.
    
    New features focus on MISSING checks rather than just counting risky constructs:
    - dangerous_call_without_check_count/ratio
    - pointer_deref_without_null_check_count/ratio
    - array_access_without_bounds_check_count/ratio
    - malloc_without_free_count/ratio
    - free_without_null_check_count/ratio
    - unchecked_return_value_count/ratio
    - defense_ratio (defensive checks / risky operations)
    """
    # Initialize all features with zeros
    features: Dict[str, float] = {
        'loc': 0.0,
        'stmt_count': 0.0,
        # Dangerous calls
        'dangerous_call_count': 0.0,
        'dangerous_call_without_check_count': 0.0,
        'dangerous_call_without_check_ratio': 0.0,
        # Pointer derefs
        'pointer_deref_count': 0.0,
        'pointer_deref_without_null_check_count': 0.0,
        'pointer_deref_without_null_check_ratio': 0.0,
        # Array access
        'array_access_count': 0.0,
        'array_access_without_bounds_check_count': 0.0,
        'array_access_without_bounds_check_ratio': 0.0,
        # Memory management
        'malloc_count': 0.0,
        'malloc_without_free_count': 0.0,
        'malloc_without_free_ratio': 0.0,
        'free_count': 0.0,
        'free_without_null_check_count': 0.0,
        'free_without_null_check_ratio': 0.0,
        # Unchecked returns
        'unchecked_return_value_count': 0.0,
        'unchecked_return_value_ratio': 0.0,
        # Defensive features
        'null_check_count': 0.0,
        'bounds_check_count': 0.0,
        'defense_ratio': 0.0,
        # Densities (per LOC)
        'dangerous_call_density': 0.0,
        'pointer_deref_density': 0.0,
        'array_access_density': 0.0,
        'null_check_density': 0.0,
    }
    
    if not code or not code.strip():
        return features
    
    # Build graph context if not provided
    parse_result, cfg, dfg = _build_graph_context(code, parse_result, cfg, dfg)
    
    # Basic size metrics
    size_metrics = _compute_basic_size_metrics(code)
    features.update(size_metrics)
    loc = max(size_metrics['loc'], 1.0)
    
    # Index conditions and checks (for path-based analysis)
    null_checked_vars, bounds_checked_vars, condition_lines = _index_conditions_and_checks(
        code, parse_result
    )
    
    # Compute each feature family
    # 1. Dangerous calls
    dc_features = _compute_dangerous_call_features(
        code, dictionary, parse_result, cfg, dfg, null_checked_vars, bounds_checked_vars
    )
    features.update(dc_features)
    
    # 2. Pointer derefs
    pd_features = _compute_pointer_deref_features(
        code, dfg, cfg, null_checked_vars
    )
    features.update(pd_features)
    
    # 3. Array access
    aa_features = _compute_array_access_features(
        code, parse_result, cfg, bounds_checked_vars
    )
    features.update(aa_features)
    
    # 4. Malloc/free patterns
    mf_features = _compute_malloc_free_features(
        code, parse_result, dfg, null_checked_vars
    )
    features.update(mf_features)
    
    # 5. Unchecked return values
    ur_features = _compute_unchecked_return_features(
        code, parse_result, dictionary
    )
    features.update(ur_features)
    
    # 6. Defense ratio and densities
    features['null_check_count'] = float(len(null_checked_vars))
    features['bounds_check_count'] = float(len(bounds_checked_vars))
    
    risk_ops = (features['dangerous_call_count'] + features['pointer_deref_count'] + 
                features['array_access_count'] + features['malloc_count'])
    defense_count = features['null_check_count'] + features['bounds_check_count']
    features['defense_ratio'] = defense_count / max(risk_ops, 1.0)
    
    # Densities
    features['dangerous_call_density'] = features['dangerous_call_count'] / loc
    features['pointer_deref_density'] = features['pointer_deref_count'] / loc
    features['array_access_density'] = features['array_access_count'] / loc
    features['null_check_density'] = features['null_check_count'] / loc
    
    return features
