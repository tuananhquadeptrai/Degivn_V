"""Model inference module for HierarchicalBiGRU vulnerability detection.

This module matches the training pipeline from 03_training_v2.py.
Extended with attention-based vulnerability localization and graph-aware slicing.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Set
from dataclasses import dataclass, field
import logging

import torch
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from devign_pipeline.src.models.hierarchical_bigru import HierarchicalBiGRU, AttentionWeights

# Graph-aware imports
try:
    from devign_pipeline.src.ast.parser import CFamilyParser, ParseResult
    from devign_pipeline.src.graphs.cfg import CFGBuilder, CFG
    from devign_pipeline.src.graphs.dfg import DFGBuilder, DFG
    from devign_pipeline.src.slicing.slicer import CodeSlicer, SliceConfig, SliceType, CodeSlice
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    CFamilyParser = None
    ParseResult = None
    CFGBuilder = None
    CFG = None
    DFGBuilder = None
    DFG = None
    CodeSlicer = None
    SliceConfig = None
    SliceType = None
    CodeSlice = None

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = [
    Path("models/best_v2_seed42.pt"),
    Path("models/best_v2_seed1042.pt"),
    Path("models/best_v2_seed2042.pt"),
]
VOCAB_PATH = Path("models/vocab.json")
CONFIG_PATH = Path("models/config.json")
FEATURE_STATS_PATH = Path("models/feature_stats.json")
ENSEMBLE_CONFIG_PATH = Path("ensemble_config.json")
CALIBRATOR_PATH = Path("models/calibrator.joblib")

MAX_LEN = 512
NUM_SLICES = 6
SLICE_LEN = 256
VULN_DIM = 26


class PredictionRequest(BaseModel):
    code: str


class PredictionResponse(BaseModel):
    vulnerable: bool
    score: float
    threshold: float
    confidence: str
    detected_patterns: List[str] = []


@dataclass
class VulnerableLocation:
    """A highlighted vulnerable location in the code."""
    line: int
    score: float
    normalized_score: float
    code_snippet: str = ""
    tokens: List[str] = field(default_factory=list)


@dataclass 
class LocalizationResult:
    """Full localization result with prediction and highlights."""
    prediction: PredictionResponse
    highlights: List[VulnerableLocation]
    

C_KEYWORDS = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
    'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
    'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while',
    'NULL', 'true', 'false', 'nullptr'
}

COMMON_STDLIB_FUNCS = {
    'printf', 'scanf', 'malloc', 'calloc', 'realloc', 'free',
    'memcpy', 'memset', 'memmove', 'memcmp', 'strlen', 'strcpy',
    'strncpy', 'strcat', 'strncat', 'strcmp', 'strncmp', 'strchr',
    'strrchr', 'strstr', 'sprintf', 'snprintf', 'sscanf', 'fprintf',
    'fscanf', 'fopen', 'fclose', 'fread', 'fwrite', 'fgets', 'fputs',
    'exit', 'abort', 'atoi', 'atol', 'atof', 'strtol', 'getchar', 'putchar',
    'gets', 'puts', 'getenv', 'system', 'assert', 'perror', 'open', 'close',
    'read', 'write', 'alloca', 'vprintf', 'vsnprintf', 'vsprintf'
}

DANGEROUS_FUNCTIONS = [
    'strcpy', 'strcat', 'gets', 'sprintf', 'memcpy', 'memmove',
    'scanf', 'fscanf', 'sscanf', 'vsprintf', 'vprintf'
]

# Dangerous sinks for taint analysis (truly unsafe functions only)
DANGEROUS_SINKS = {
    'strcpy', 'strcat', 'gets', 'sprintf', 'memcpy', 'memmove',
    'scanf', 'fscanf', 'sscanf', 'vsprintf', 'vprintf'
}

# Safe alternatives (have bounds checking)
SAFE_ALTERNATIVES = {
    'strncpy', 'strncat', 'snprintf', 'fgets', 'vsnprintf'
}

# External input sources
TAINT_SOURCES = {
    'read', 'fread', 'fgets', 'gets', 'getchar', 'scanf', 'fscanf',
    'sscanf', 'getenv', 'recv', 'recvfrom', 'recvmsg', 'fgetc'
}


def detect_critical_patterns(code: str) -> Tuple[float, List[str]]:
    """Detect critical vulnerability patterns using static analysis.
    
    Returns:
        Tuple of (boost_score, list_of_detected_patterns)
    """
    boost_score = 0.0
    detected_patterns = []
    
    # 1. gets() - buffer overflow (boost +0.15)
    if re.search(r'\bgets\s*\(', code):
        boost_score += 0.15
        detected_patterns.append("gets() - buffer overflow risk")
    
    # 2. printf/sprintf/fprintf with user input - format string (boost +0.12)
    # Detect printf(var) or printf(buf) without format string literal
    format_funcs = ['printf', 'sprintf', 'fprintf', 'vprintf', 'vsprintf']
    for func in format_funcs:
        # Match func(non-string-literal) - potential format string vuln
        pattern = rf'\b{func}\s*\(\s*([^",\)]+)\s*\)'
        matches = re.findall(pattern, code)
        for match in matches:
            # If argument is not a string literal, it's potentially dangerous
            if not match.strip().startswith('"'):
                boost_score += 0.12
                detected_patterns.append(f"{func}(user_input) - format string vulnerability")
                break
    
    # 3. strcpy/strcat without bounds check (boost +0.10)
    for func in ['strcpy', 'strcat']:
        if re.search(rf'\b{func}\s*\(', code):
            # Check if strncpy/strncat is also used (indicates awareness of bounds)
            safe_version = func.replace('cpy', 'ncpy').replace('cat', 'ncat')
            if not re.search(rf'\b{safe_version}\s*\(', code):
                # Check for sizeof or length checks nearby
                if not re.search(r'\bsizeof\s*\(', code) and not re.search(r'\bstrlen\s*\(', code):
                    boost_score += 0.10
                    detected_patterns.append(f"{func}() without bounds check")
    
    # 4. malloc without NULL check (boost +0.08)
    malloc_matches = list(re.finditer(r'(\w+)\s*=\s*(\(?\s*\w+\s*\*?\s*\)?)\s*malloc\s*\(', code))
    for match in malloc_matches:
        var_name = match.group(1)
        # Look for NULL check after malloc
        after_malloc = code[match.end():]
        null_check_pattern = rf'\bif\s*\(\s*{re.escape(var_name)}\s*(==\s*NULL|!=\s*NULL|!|==\s*0|!=\s*0)'
        if not re.search(null_check_pattern, after_malloc[:200]):
            boost_score += 0.08
            detected_patterns.append(f"malloc() without NULL check for '{var_name}'")
            break
    
    # 5. Double free detection (boost +0.15)
    free_calls = list(re.finditer(r'\bfree\s*\(\s*(\w+)\s*\)', code))
    freed_vars = {}
    for match in free_calls:
        var_name = match.group(1)
        if var_name in freed_vars:
            # Check if the variable was reassigned between frees
            between_code = code[freed_vars[var_name]:match.start()]
            if not re.search(rf'\b{re.escape(var_name)}\s*=', between_code):
                boost_score += 0.15
                detected_patterns.append(f"double free on '{var_name}'")
                break
        freed_vars[var_name] = match.end()
    
    # 6. Use after free - return freed memory (boost +0.15)
    for match in free_calls:
        var_name = match.group(1)
        after_free = code[match.end():]
        # Check if variable is nullified after free (safe pattern)
        # Match both: var = NULL and *var = NULL
        if re.search(rf'(\*\s*)?{re.escape(var_name)}\s*=\s*NULL\b', after_free[:100]):
            continue
        # Check for immediate return NULL after free (safe cleanup pattern)
        if re.search(r'^\s*;\s*\n\s*return\s+NULL\s*;', after_free[:50]):
            continue
        # Check if variable is used after free (not reassigned)
        if not re.search(rf'\b{re.escape(var_name)}\s*=', after_free[:100]):
            # Check if returned or dereferenced
            if re.search(rf'\breturn\s+{re.escape(var_name)}\b', after_free[:200]):
                boost_score += 0.15
                detected_patterns.append(f"use after free - returning freed '{var_name}'")
                break
            # Check for pointer dereference after free (exclude *var = NULL pattern)
            if re.search(rf'\b{re.escape(var_name)}\s*(\[|->)', after_free[:200]):
                boost_score += 0.15
                detected_patterns.append(f"use after free - accessing freed '{var_name}'")
                break
    
    return (boost_score, detected_patterns)


@dataclass
class GraphAnalysis:
    """Results from AST/CFG/DFG analysis for vulnerability validation."""
    has_dangerous_flow: bool = False
    dangerous_call_count: int = 0
    unguarded_dangerous_calls: int = 0
    tainted_sinks: List[Tuple[str, int]] = field(default_factory=list)  # (func_name, line)
    missing_null_checks: int = 0
    missing_bounds_checks: int = 0
    risk_score: float = 0.0
    analysis_success: bool = False


@dataclass
class TokenInfo:
    """Token with position information."""
    text: str
    line: int
    start_pos: int
    end_pos: int


def strip_comments(code: str) -> str:
    """Remove all C/C++ comments completely from code.
    
    This removes both single-line (//) and multi-line (/* */) comments
    to avoid comment bias in vulnerability detection.
    """
    # Remove multi-line comments /* */ completely
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove single-line comments // completely  
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    return code


def tokenize_c_code_with_positions(code: str) -> Tuple[List[str], List[int]]:
    """Tokenize C code and return tokens with their line numbers.
    
    Comments are completely stripped before tokenization to avoid
    comment bias in the model (e.g., comments mentioning 'vulnerability'
    should not affect detection).
    """
    # Strip comments completely before processing
    clean_code = strip_comments(code)
    
    lines = clean_code.split('\n')
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line) + 1)
    
    def pos_to_line(pos: int) -> int:
        for i, start in enumerate(line_starts):
            if i + 1 < len(line_starts) and pos < line_starts[i + 1]:
                return i + 1
        return len(lines)
    
    patterns = [
        (r'"(?:[^"\\]|\\.)*"', 'STR'),
        (r"'(?:[^'\\]|\\.)*'", 'CHAR'),
        (r'0[xX][0-9a-fA-F]+[uUlL]*', 'NUM'),
        (r'0[bB][01]+[uUlL]*', 'NUM'),
        (r'\d+\.?\d*(?:[eE][+-]?\d+)?[fFlLuU]*', 'NUM'),
        (r'\.\d+(?:[eE][+-]?\d+)?[fFlL]*', 'NUM'),
        (r'\.\.\.', None),
        (r'::', None),
        (r'->', None),
        (r'\+\+|--', None),
        (r'<<=|>>=', None),
        (r'<<|>>', None),
        (r'<=|>=|==|!=', None),
        (r'&&|\|\|', None),
        (r'[+\-*/%&|^~!=<>]=', None),
        (r'[+\-*/%&|^~!=<>?:#]', None),
        (r'[a-zA-Z_][a-zA-Z0-9_]*', 'ID'),
        (r'[{}()\[\];,.]', None),
    ]
    
    compiled = [(re.compile(p), t) for p, t in patterns]
    
    tokens = []
    token_lines = []
    pos = 0
    
    while pos < len(clean_code):
        if clean_code[pos].isspace():
            pos += 1
            continue
        
        matched = False
        for pattern, token_type in compiled:
            match = pattern.match(clean_code, pos)
            if match:
                text = match.group()
                line = pos_to_line(pos)
                
                if token_type == 'STR':
                    tokens.append('STR')
                elif token_type == 'CHAR':
                    tokens.append('CHAR')
                elif token_type == 'NUM':
                    tokens.append('NUM')
                else:
                    tokens.append(text)
                
                token_lines.append(line)
                pos = match.end()
                matched = True
                break
        
        if not matched:
            pos += 1
    
    return tokens, token_lines


def normalize_tokens(tokens: List[str]) -> List[str]:
    """Normalize tokens: variables -> VAR_N, non-stdlib functions -> FUNC_N."""
    normalized = []
    var_map: Dict[str, str] = {}
    func_map: Dict[str, str] = {}
    var_counter = 0
    func_counter = 0
    
    for i, token in enumerate(tokens):
        if token in C_KEYWORDS:
            normalized.append(token)
        elif token in ('NUM', 'STR', 'CHAR'):
            normalized.append(token)
        elif token in COMMON_STDLIB_FUNCS:
            normalized.append(token)
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None
            is_func = next_token == '('
            
            if is_func:
                if token not in func_map:
                    func_map[token] = f'FUNC_{func_counter}'
                    func_counter += 1
                normalized.append(func_map[token])
            else:
                if token not in var_map:
                    var_map[token] = f'VAR_{var_counter}'
                    var_counter += 1
                normalized.append(var_map[token])
        else:
            normalized.append(token)
    
    return normalized


def extract_vuln_features(code: str, tokens: List[str]) -> np.ndarray:
    """Extract 26 vulnerability features matching training."""
    feats = []
    
    loc = code.count('\n') + 1
    feats.append(loc)
    
    stmt_count = code.count(';')
    feats.append(stmt_count)
    
    dangerous_count = 0
    for func in DANGEROUS_FUNCTIONS:
        dangerous_count += len(re.findall(rf'\b{func}\s*\(', code))
    feats.append(dangerous_count)
    
    if_count = len(re.findall(r'\bif\s*\(', code))
    feats.append(max(0, dangerous_count - if_count))
    
    feats.append(dangerous_count / max(1, stmt_count) if stmt_count else 0)
    
    ptr_deref = code.count('->') + code.count('*')
    feats.append(ptr_deref)
    
    null_checks = len(re.findall(r'\bNULL\b', code)) + len(re.findall(r'!=\s*NULL', code))
    feats.append(max(0, ptr_deref - null_checks))
    
    feats.append(max(0, ptr_deref - null_checks) / max(1, ptr_deref) if ptr_deref else 0)
    
    array_access = code.count('[')
    feats.append(array_access)
    
    bounds_checks = len(re.findall(r'<\s*\w+', code)) + len(re.findall(r'>\s*0', code))
    feats.append(max(0, array_access - bounds_checks))
    
    feats.append(max(0, array_access - bounds_checks) / max(1, array_access) if array_access else 0)
    
    malloc_count = len(re.findall(r'\bmalloc\s*\(', code)) + len(re.findall(r'\bcalloc\s*\(', code))
    feats.append(malloc_count)
    
    free_count = len(re.findall(r'\bfree\s*\(', code))
    feats.append(max(0, malloc_count - free_count))
    
    feats.append(max(0, malloc_count - free_count) / max(1, malloc_count) if malloc_count else 0)
    
    feats.append(free_count)
    
    feats.append(max(0, free_count - null_checks))
    
    feats.append(max(0, free_count - null_checks) / max(1, free_count) if free_count else 0)
    
    func_calls = len(re.findall(r'\b\w+\s*\(', code))
    feats.append(max(0, func_calls - if_count))
    
    feats.append(max(0, func_calls - if_count) / max(1, func_calls) if func_calls else 0)
    
    feats.append(null_checks)
    
    feats.append(bounds_checks)
    
    feats.append((null_checks + bounds_checks) / max(1, dangerous_count + ptr_deref + array_access))
    
    feats.append(dangerous_count / max(1, loc))
    
    feats.append(ptr_deref / max(1, loc))
    
    feats.append(array_access / max(1, loc))
    
    feats.append(null_checks / max(1, loc))
    
    if len(feats) < VULN_DIM:
        feats.extend([0.0] * (VULN_DIM - len(feats)))
    else:
        feats = feats[:VULN_DIM]
    
    return np.array(feats, dtype=np.float32)


class GraphAnalyzer:
    """Analyze code using AST/CFG/DFG for vulnerability validation and slicing."""
    
    def __init__(self):
        if not GRAPH_AVAILABLE:
            raise RuntimeError("Graph analysis requires tree-sitter packages")
        self.parser = CFamilyParser()
        self.cfg_builder = CFGBuilder()
        self.dfg_builder = DFGBuilder(build_full_function=True)
        
    def parse_code(self, code: str) -> Optional[ParseResult]:
        """Parse C code into AST."""
        try:
            return self.parser.parse_with_fallback(code)
        except Exception as e:
            logger.debug(f"Parse failed: {e}")
            return None
    
    def build_cfg(self, parse_result: ParseResult) -> Optional[CFG]:
        """Build Control Flow Graph from parsed AST."""
        try:
            return self.cfg_builder.build(parse_result)
        except Exception as e:
            logger.debug(f"CFG build failed: {e}")
            return None
    
    def build_dfg(self, parse_result: ParseResult, focus_lines: List[int] = None) -> Optional[DFG]:
        """Build Data Flow Graph from parsed AST."""
        try:
            return self.dfg_builder.build(parse_result, focus_lines)
        except Exception as e:
            logger.debug(f"DFG build failed: {e}")
            return None
    
    def analyze(self, code: str) -> GraphAnalysis:
        """Perform full graph analysis for vulnerability validation."""
        result = GraphAnalysis()
        
        parse_result = self.parse_code(code)
        if not parse_result or not parse_result.nodes:
            return result
        
        result.analysis_success = True
        cfg = self.build_cfg(parse_result)
        dfg = self.build_dfg(parse_result)
        
        # Count dangerous function calls
        dangerous_calls = self._find_dangerous_calls(code, parse_result)
        result.dangerous_call_count = len(dangerous_calls)
        
        # Find unguarded dangerous calls (no prior if-check on same line or nearby)
        result.unguarded_dangerous_calls = self._count_unguarded_calls(
            code, dangerous_calls, parse_result, cfg
        )
        
        # Find tainted flows (source -> sink without sanitization)
        if dfg:
            result.tainted_sinks = self._find_tainted_sinks(parse_result, dfg)
            result.has_dangerous_flow = len(result.tainted_sinks) > 0
        
        # Analyze pointer/array safety
        result.missing_null_checks = self._count_missing_null_checks(code, parse_result)
        result.missing_bounds_checks = self._count_missing_bounds_checks(code, parse_result)
        
        # Calculate risk score
        result.risk_score = self._calculate_risk_score(result)
        
        return result
    
    def _find_dangerous_calls(self, code: str, parse_result: ParseResult) -> List[Tuple[str, int]]:
        """Find dangerous function calls with line numbers."""
        calls = []
        for node in parse_result.nodes:
            if node.node_type == 'call_expression':
                for child_idx in node.children_indices:
                    child = parse_result.nodes[child_idx]
                    if child.node_type == 'identifier':
                        func_name = child.text.strip()
                        if func_name in DANGEROUS_SINKS:
                            calls.append((func_name, node.start_line))
        return calls
    
    def _count_unguarded_calls(
        self, 
        code: str, 
        dangerous_calls: List[Tuple[str, int]], 
        parse_result: ParseResult,
        cfg: Optional[CFG]
    ) -> int:
        """Count dangerous calls not preceded by relevant checks."""
        if not dangerous_calls:
            return 0
        
        unguarded = 0
        lines = code.split('\n')
        
        for func_name, line in dangerous_calls:
            # Check if there's a null check or bounds check in preceding lines
            has_check = False
            check_window = 5  # Look back 5 lines for checks
            
            for i in range(max(0, line - check_window - 1), line - 1):
                if i < len(lines):
                    line_content = lines[i]
                    # Check for NULL check
                    if 'NULL' in line_content or '!= 0' in line_content:
                        has_check = True
                        break
                    # Check for bounds check
                    if '<' in line_content and ('[' in line_content or 'len' in line_content.lower()):
                        has_check = True
                        break
            
            # Also check if call is inside an if block
            if cfg:
                block_ids = cfg.get_blocks_for_lines([line])
                for block_id in block_ids:
                    preds = cfg.get_predecessors(block_id)
                    for pred_id in preds:
                        pred_block = cfg.get_block_by_id(pred_id)
                        if pred_block and pred_block.block_type == 'condition':
                            has_check = True
                            break
            
            if not has_check:
                unguarded += 1
        
        return unguarded
    
    def _find_tainted_sinks(self, parse_result: ParseResult, dfg: DFG) -> List[Tuple[str, int]]:
        """Find sinks that receive data from tainted sources."""
        tainted_sinks = []
        
        # Find source nodes
        source_vars: Set[str] = set()
        for node in dfg.nodes:
            if 'call:' in node.context:
                func_name = node.context.split(':')[1]
                if func_name in TAINT_SOURCES:
                    source_vars.add(node.var_name)
        
        # Also mark function parameters as potentially tainted
        for node in dfg.nodes:
            if node.context == 'parameter':
                source_vars.add(node.var_name)
        
        if not source_vars:
            return tainted_sinks
        
        # Track taint propagation
        tainted = set(source_vars)
        changed = True
        max_iterations = 10
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for edge in dfg.edges:
                if edge.edge_type == 'def-use':
                    from_node = dfg.nodes[edge.from_idx]
                    to_node = dfg.nodes[edge.to_idx]
                    
                    if from_node.var_name in tainted and to_node.var_name not in tainted:
                        tainted.add(to_node.var_name)
                        changed = True
        
        # Find sinks using tainted data
        for node in dfg.nodes:
            if 'call:' in node.context:
                func_name = node.context.split(':')[1]
                if func_name in DANGEROUS_SINKS and node.var_name in tainted:
                    tainted_sinks.append((func_name, node.line))
        
        return tainted_sinks
    
    def _count_missing_null_checks(self, code: str, parse_result: ParseResult) -> int:
        """Count pointer dereferences without null checks."""
        ptr_deref_count = 0
        null_check_count = 0
        
        for node in parse_result.nodes:
            if node.node_type == 'pointer_expression' and node.text.startswith('*'):
                ptr_deref_count += 1
        
        null_check_count = len(re.findall(r'!=\s*NULL|==\s*NULL|!\s*\w+', code))
        
        return max(0, ptr_deref_count - null_check_count)
    
    def _count_missing_bounds_checks(self, code: str, parse_result: ParseResult) -> int:
        """Count array accesses without bounds checks."""
        array_access_count = 0
        bounds_check_count = 0
        
        for node in parse_result.nodes:
            if node.node_type == 'subscript_expression':
                array_access_count += 1
        
        bounds_check_count = len(re.findall(r'<\s*\w+\s*(?:;|\))|sizeof|strlen|len', code))
        
        return max(0, array_access_count - bounds_check_count)
    
    def _calculate_risk_score(self, analysis: GraphAnalysis) -> float:
        """Calculate overall risk score from analysis results."""
        score = 0.0
        
        # Tainted sinks are high risk
        score += len(analysis.tainted_sinks) * 0.3
        
        # Unguarded dangerous calls
        score += analysis.unguarded_dangerous_calls * 0.2
        
        # Missing checks
        score += analysis.missing_null_checks * 0.1
        score += analysis.missing_bounds_checks * 0.1
        
        # Cap at 1.0
        return min(1.0, score)
    
    def get_dependency_slices(
        self, 
        code: str, 
        num_slices: int = 6, 
        slice_len: int = 256
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """Get dependency-aware slices centered on dangerous operations."""
        parse_result = self.parse_code(code)
        if not parse_result:
            return self._fallback_even_slices(code, num_slices, slice_len)
        
        cfg = self.build_cfg(parse_result)
        dfg = self.build_dfg(parse_result)
        
        # Find seed lines (dangerous operations)
        seed_lines = self._find_seed_lines(code, parse_result)
        
        if not seed_lines:
            return self._fallback_even_slices(code, num_slices, slice_len)
        
        # Create slicer with graph-aware config
        config = SliceConfig(
            slice_type=SliceType.BACKWARD,
            window_size=15,
            include_control_deps=True,
            include_data_deps=True,
            max_depth=5,
            use_post_dominator=True,
        )
        slicer = CodeSlicer(config)
        
        slices_data = []
        
        # Generate slices for each seed (up to num_slices)
        for seed_line in seed_lines[:num_slices]:
            try:
                backward_slice = slicer.backward_slice(code, [seed_line], cfg, dfg)
                
                # Tokenize slice content
                tokens, token_lines = tokenize_c_code_with_positions(backward_slice.code)
                normalized = normalize_tokens(tokens)
                
                slices_data.append({
                    'tokens': normalized[:slice_len - 2],
                    'token_lines': token_lines[:slice_len - 2],
                    'seed_line': seed_line,
                    'included_lines': list(backward_slice.included_lines),
                })
            except Exception:
                continue
        
        # If we don't have enough slices, add forward slices
        if len(slices_data) < num_slices:
            config.slice_type = SliceType.FORWARD
            forward_slicer = CodeSlicer(config)
            
            for seed_line in seed_lines:
                if len(slices_data) >= num_slices:
                    break
                try:
                    forward_slice = forward_slicer.forward_slice(code, [seed_line], cfg, dfg)
                    tokens, token_lines = tokenize_c_code_with_positions(forward_slice.code)
                    normalized = normalize_tokens(tokens)
                    
                    slices_data.append({
                        'tokens': normalized[:slice_len - 2],
                        'token_lines': token_lines[:slice_len - 2],
                        'seed_line': seed_line,
                        'included_lines': list(forward_slice.included_lines),
                    })
                except Exception:
                    continue
        
        # Fill remaining with fallback
        if len(slices_data) < num_slices:
            fallback_tokens, fallback_lines = self._fallback_even_slices(code, num_slices - len(slices_data), slice_len)
            for i in range(len(fallback_tokens)):
                slices_data.append({
                    'tokens': fallback_tokens[i],
                    'token_lines': fallback_lines[i] if i < len(fallback_lines) else [],
                })
        
        # Extract result
        result_tokens = []
        result_lines = []
        for i in range(num_slices):
            if i < len(slices_data):
                result_tokens.append(slices_data[i].get('tokens', []))
                result_lines.append(slices_data[i].get('token_lines', []))
            else:
                result_tokens.append([])
                result_lines.append([])
        
        return result_tokens, result_lines
    
    def _find_seed_lines(self, code: str, parse_result: ParseResult) -> List[int]:
        """Find seed lines for slicing (dangerous operations, memory ops, etc.)."""
        seed_lines = []
        seed_priorities = []  # (line, priority) - lower is higher priority
        
        for node in parse_result.nodes:
            # Dangerous function calls - highest priority
            if node.node_type == 'call_expression':
                for child_idx in node.children_indices:
                    child = parse_result.nodes[child_idx]
                    if child.node_type == 'identifier':
                        func_name = child.text.strip()
                        if func_name in DANGEROUS_SINKS:
                            seed_priorities.append((node.start_line, 1))
                        elif func_name in {'malloc', 'calloc', 'realloc', 'free'}:
                            seed_priorities.append((node.start_line, 2))
            
            # Pointer dereferences - medium priority
            elif node.node_type == 'pointer_expression':
                seed_priorities.append((node.start_line, 3))
            
            # Array accesses - medium priority
            elif node.node_type == 'subscript_expression':
                seed_priorities.append((node.start_line, 3))
        
        # Sort by priority and deduplicate
        seed_priorities.sort(key=lambda x: (x[1], x[0]))
        seen = set()
        for line, _ in seed_priorities:
            if line not in seen:
                seed_lines.append(line)
                seen.add(line)
        
        return seed_lines
    
    def _fallback_even_slices(
        self, 
        code: str, 
        num_slices: int, 
        slice_len: int
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """Fallback to even slicing when graph analysis fails."""
        tokens, token_lines = tokenize_c_code_with_positions(code)
        normalized = normalize_tokens(tokens)
        
        result_tokens = []
        result_lines = []
        
        for start in range(0, len(normalized), slice_len - 2):
            if len(result_tokens) >= num_slices:
                break
            end = min(start + slice_len - 2, len(normalized))
            result_tokens.append(normalized[start:end])
            result_lines.append(token_lines[start:end])
        
        while len(result_tokens) < num_slices:
            result_tokens.append([])
            result_lines.append([])
        
        return result_tokens, result_lines


class ModelWrapper:
    def __init__(
        self,
        model_paths: List[Path] = None,
        vocab_path: Path = VOCAB_PATH,
        config_path: Path = CONFIG_PATH,
        feature_stats_path: Path = FEATURE_STATS_PATH,
        ensemble_config_path: Path = ENSEMBLE_CONFIG_PATH,
        calibrator_path: Path = CALIBRATOR_PATH,
        use_graph_slicing: bool = True,
        use_graph_postprocessing: bool = True,
    ) -> None:
        self.model_paths = model_paths or MODEL_PATHS
        self.vocab_path = vocab_path
        self.config_path = config_path
        self.feature_stats_path = feature_stats_path
        self.ensemble_config_path = ensemble_config_path
        self.calibrator_path = calibrator_path
        self.use_graph_slicing = use_graph_slicing and GRAPH_AVAILABLE
        self.use_graph_postprocessing = use_graph_postprocessing and GRAPH_AVAILABLE
        
        self.data_config = self._load_data_config()
        self.ensemble_config = self._load_ensemble_config()
        self.vocab = self._load_vocab()
        self.feature_stats = self._load_feature_stats()
        self.calibrator = self._load_calibrator()
        self.models = self._load_models()
        self.threshold = float(self.ensemble_config.get("optimal_threshold", 0.65))
        
        self.max_len = self.data_config.get("max_len", MAX_LEN)
        self.num_slices = self.data_config.get("max_slices", NUM_SLICES)
        self.slice_len = self.data_config.get("slice_max_len", SLICE_LEN)
        
        # Initialize graph analyzer if available
        self.graph_analyzer: Optional[GraphAnalyzer] = None
        if GRAPH_AVAILABLE and (use_graph_slicing or use_graph_postprocessing):
            try:
                self.graph_analyzer = GraphAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to initialize GraphAnalyzer: {e}")
                self.use_graph_slicing = False
                self.use_graph_postprocessing = False

    def _load_data_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"vocab_size": 238, "max_len": 512, "max_slices": 6, "slice_max_len": 256}

    def _load_ensemble_config(self) -> Dict[str, Any]:
        if self.ensemble_config_path.exists():
            return json.loads(self.ensemble_config_path.read_text())
        return {"optimal_threshold": 0.65}

    def _load_vocab(self) -> Dict[str, int]:
        if self.vocab_path.exists():
            data = json.loads(self.vocab_path.read_text())
            return data.get("token2id", {})
        return {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    
    def _load_feature_stats(self) -> Dict[str, Any]:
        if self.feature_stats_path.exists():
            data = json.loads(self.feature_stats_path.read_text())
            return data.get("feature_stats", {})
        return {}
    
    def _load_calibrator(self) -> Optional[Any]:
        """Load isotonic calibrator if available."""
        if self.calibrator_path.exists():
            try:
                import joblib
                calibrator = joblib.load(self.calibrator_path)
                logger.info(f"Loaded isotonic calibrator from {self.calibrator_path}")
                return calibrator
            except Exception as e:
                logger.warning(f"Failed to load calibrator: {e}")
        return None
    
    def _calibrate_probability(self, prob: float) -> float:
        """Apply isotonic calibration to raw probability."""
        if self.calibrator is not None:
            try:
                calibrated = self.calibrator.predict([prob])[0]
                return float(calibrated)
            except Exception as e:
                logger.debug(f"Calibration failed, using raw prob: {e}")
        return prob

    def _load_single_model(self, model_path: Path) -> HierarchicalBiGRU:
        """Load a single model from path."""
        vocab_size = self.data_config.get("vocab_size", 238)
        
        model = HierarchicalBiGRU(
            vocab_size=vocab_size,
            embed_dim=96,
            hidden_dim=192,
            slice_hidden=160,
            vuln_dim=VULN_DIM,
            slice_feat_dim=52,
            gate_init=float(self.ensemble_config.get("gate_init", 0.4)),
        )
        
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        
        model.to(DEVICE)
        model.eval()
        return model

    def _load_models(self) -> List[HierarchicalBiGRU]:
        """Load ensemble of models."""
        models = []
        for path in self.model_paths:
            if path.exists():
                models.append(self._load_single_model(path))
        
        if not models:
            models.append(self._load_single_model(self.model_paths[0]))
        
        return models

    def _preprocess_with_mapping(self, code: str) -> Tuple[Dict[str, torch.Tensor], List[str], List[int], List[int], List[Tuple[int, int]]]:
        """Preprocess and return token-to-position mappings for localization.
        
        Now supports graph-aware slicing when available.
        """
        tokens, token_lines = tokenize_c_code_with_positions(code)
        normalized = normalize_tokens(tokens)
        
        pad_id = self.vocab.get("<PAD>", 0)
        unk_id = self.vocab.get("<UNK>", 1)
        bos_id = self.vocab.get("<BOS>", 2)
        eos_id = self.vocab.get("<EOS>", 3)
        
        # Global sequence encoding (unchanged)
        global_ids = [bos_id]
        global_token_indices = []
        for i, t in enumerate(normalized[:self.max_len - 2]):
            global_ids.append(self.vocab.get(t, unk_id))
            global_token_indices.append(i)
        global_ids.append(eos_id)
        
        input_ids = torch.full((1, self.max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((1, self.max_len), dtype=torch.float)
        length = min(len(global_ids), self.max_len)
        input_ids[0, :length] = torch.tensor(global_ids[:length], dtype=torch.long)
        attention_mask[0, :length] = 1.0
        
        # Slice encoding - use graph-aware slicing if enabled
        slices = []
        slice_token_indices = []
        
        if self.use_graph_slicing and self.graph_analyzer:
            try:
                # Get dependency-aware slices
                graph_slices, graph_lines = self.graph_analyzer.get_dependency_slices(
                    code, self.num_slices, self.slice_len
                )
                
                for slice_tokens in graph_slices:
                    slices.append(slice_tokens)
                    # Map back to original token indices (approximate)
                    indices = []
                    for tok in slice_tokens:
                        for i, orig_tok in enumerate(normalized):
                            if orig_tok == tok and i not in sum(slice_token_indices, []):
                                indices.append(i)
                                break
                    slice_token_indices.append(indices)
                    
            except Exception as e:
                logger.debug(f"Graph slicing failed, using even slicing: {e}")
                slices = []
                slice_token_indices = []
        
        # Fallback to even slicing
        if not slices:
            for start in range(0, len(normalized), self.slice_len - 2):
                if len(slices) >= self.num_slices:
                    break
                end = min(start + self.slice_len - 2, len(normalized))
                slices.append(normalized[start:end])
                slice_token_indices.append(list(range(start, end)))
        
        while len(slices) < self.num_slices:
            slices.append([])
            slice_token_indices.append([])
        
        slice_input_ids = torch.full((1, self.num_slices, self.slice_len), pad_id, dtype=torch.long)
        slice_attention_mask = torch.zeros((1, self.num_slices, self.slice_len), dtype=torch.float)
        valid_slices = 0
        
        slice_pos_mapping: List[Tuple[int, int]] = [(-1, -1)] * len(normalized)
        
        for s_idx, (slice_tokens, token_indices) in enumerate(zip(slices, slice_token_indices)):
            if not slice_tokens:
                continue
            
            ids = [bos_id]
            for j, t in enumerate(slice_tokens[:self.slice_len - 2]):
                ids.append(self.vocab.get(t, unk_id))
                if j < len(token_indices):
                    tok_idx = token_indices[j]
                    if tok_idx < len(slice_pos_mapping):
                        slice_pos_mapping[tok_idx] = (s_idx, j + 1)
            ids.append(eos_id)
            
            length = min(len(ids), self.slice_len)
            slice_input_ids[0, s_idx, :length] = torch.tensor(ids[:length], dtype=torch.long)
            slice_attention_mask[0, s_idx, :length] = 1.0
            valid_slices += 1
        
        slice_count = torch.tensor([max(1, valid_slices)], dtype=torch.long)
        
        vuln_features = self._extract_and_normalize_features(code, tokens)
        
        slice_vuln_features = torch.zeros((1, self.num_slices, VULN_DIM), dtype=torch.float)
        slice_rel_features = torch.zeros((1, self.num_slices, VULN_DIM), dtype=torch.float)
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "slice_input_ids": slice_input_ids,
            "slice_attention_mask": slice_attention_mask,
            "slice_count": slice_count,
            "vuln_features": vuln_features,
            "slice_vuln_features": slice_vuln_features,
            "slice_rel_features": slice_rel_features,
        }
        
        return inputs, tokens, token_lines, global_token_indices, slice_pos_mapping
    
    def _preprocess(self, code: str) -> Dict[str, torch.Tensor]:
        """Preprocess C code into model input tensors matching training format."""
        inputs, _, _, _, _ = self._preprocess_with_mapping(code)
        return inputs
    
    def _extract_and_normalize_features(self, code: str, tokens: List[str]) -> torch.Tensor:
        """Extract and normalize vulnerability features."""
        feats = extract_vuln_features(code, tokens)
        
        if self.feature_stats:
            feature_names = [
                "loc", "stmt_count", "dangerous_call_count",
                "dangerous_call_without_check_count", "dangerous_call_without_check_ratio",
                "pointer_deref_count", "pointer_deref_without_null_check_count",
                "pointer_deref_without_null_check_ratio", "array_access_count",
                "array_access_without_bounds_check_count", "array_access_without_bounds_check_ratio",
                "malloc_count", "malloc_without_free_count", "malloc_without_free_ratio",
                "free_count", "free_without_null_check_count", "free_without_null_check_ratio",
                "unchecked_return_value_count", "unchecked_return_value_ratio",
                "null_check_count", "bounds_check_count", "defense_ratio",
                "dangerous_call_density", "pointer_deref_density",
                "array_access_density", "null_check_density"
            ]
            
            for i, name in enumerate(feature_names):
                if i >= len(feats):
                    break
                stats = self.feature_stats.get(name, {})
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 1.0)
                if std == 0:
                    std = 1.0
                feats[i] = (feats[i] - mean) / std
        
        return torch.from_numpy(feats).unsqueeze(0)

    def _get_confidence(self, score: float) -> str:
        if score > 0.8 or score < 0.2:
            return "high"
        elif score > 0.6 or score < 0.4:
            return "medium"
        return "low"
    
    def _compute_token_importance(
        self,
        attention: AttentionWeights,
        global_token_indices: List[int],
        slice_pos_mapping: List[Tuple[int, int]],
        num_tokens: int,
        w_global: float = 0.5,
        w_slice: float = 0.5,
    ) -> np.ndarray:
        """Combine attention layers into per-token importance scores."""
        global_alpha = attention.global_alpha[0].cpu().numpy()
        slice_token_alpha = attention.slice_token_alpha[0].cpu().numpy()
        slice_level_alpha = attention.slice_level_alpha[0].cpu().numpy()
        slice_seq_alpha = attention.slice_seq_alpha[0].cpu().numpy()
        
        importance = np.zeros(num_tokens, dtype=np.float32)
        
        for i, tok_idx in enumerate(global_token_indices):
            if tok_idx < num_tokens:
                global_pos = i + 1
                if global_pos < len(global_alpha):
                    importance[tok_idx] += w_global * global_alpha[global_pos]
        
        slice_combined = 0.5 * (slice_level_alpha + slice_seq_alpha)
        
        for tok_idx, (s_idx, slice_pos) in enumerate(slice_pos_mapping):
            if tok_idx >= num_tokens:
                break
            if s_idx >= 0 and slice_pos >= 0:
                if s_idx < len(slice_combined) and slice_pos < slice_token_alpha.shape[1]:
                    slice_weight = slice_combined[s_idx] * slice_token_alpha[s_idx, slice_pos]
                    importance[tok_idx] += w_slice * slice_weight
        
        total = importance.sum()
        if total > 0:
            importance = importance / total
        
        return importance
    
    def _aggregate_to_lines(
        self,
        importance: np.ndarray,
        token_lines: List[int],
        tokens: List[str],
        code: str,
    ) -> List[VulnerableLocation]:
        """Aggregate token importance to line-level scores."""
        line_scores: Dict[int, float] = {}
        line_tokens: Dict[int, List[str]] = {}
        
        for i, (imp, line) in enumerate(zip(importance, token_lines)):
            if line not in line_scores:
                line_scores[line] = 0.0
                line_tokens[line] = []
            line_scores[line] += imp
            if imp > 0.001 and i < len(tokens):
                line_tokens[line].append(tokens[i])
        
        if not line_scores:
            return []
        
        max_score = max(line_scores.values())
        if max_score == 0:
            return []
        
        code_lines = code.split('\n')
        
        locations = []
        for line, score in sorted(line_scores.items(), key=lambda x: -x[1]):
            norm_score = score / max_score
            snippet = code_lines[line - 1].strip() if line <= len(code_lines) else ""
            locations.append(VulnerableLocation(
                line=line,
                score=float(score),
                normalized_score=float(norm_score),
                code_snippet=snippet[:100],
                tokens=line_tokens.get(line, [])[:10],
            ))
        
        return locations

    @torch.inference_mode()
    def predict(self, code: str) -> PredictionResponse:
        """Predict using ensemble of models (average probabilities).
        
        If graph-based post-processing is enabled, it may adjust scores
        for borderline predictions based on code structure analysis.
        Static analysis patterns are used to boost detection of critical vulnerabilities.
        """
        inputs = self._preprocess(code)
        
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)
        
        probs = []
        for model in self.models:
            logits = model(**inputs)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            probs.append(prob)
        
        avg_prob = sum(probs) / len(probs)
        
        # Apply isotonic calibration
        calibrated_prob = self._calibrate_probability(avg_prob)
        
        # Apply graph-based post-processing for borderline predictions
        adjusted_prob = calibrated_prob
        graph_analysis = None
        
        if self.use_graph_postprocessing and self.graph_analyzer:
            try:
                graph_analysis = self.graph_analyzer.analyze(code)
                adjusted_prob = self._apply_graph_postprocessing(calibrated_prob, graph_analysis)
            except Exception as e:
                logger.debug(f"Graph post-processing failed: {e}")
        
        # Apply static analysis pattern detection boost
        boost_score, detected_patterns = detect_critical_patterns(code)
        if boost_score > 0:
            adjusted_prob = min(1.0, adjusted_prob + boost_score)
            logger.debug(f"Pattern boost: +{boost_score:.2f}, patterns: {detected_patterns}")
        
        vulnerable = adjusted_prob >= self.threshold
        
        return PredictionResponse(
            vulnerable=vulnerable,
            score=round(adjusted_prob, 4),
            threshold=self.threshold,
            confidence=self._get_confidence(adjusted_prob),
            detected_patterns=detected_patterns,
        )
    
    def _apply_graph_postprocessing(
        self, 
        model_prob: float, 
        graph_analysis: GraphAnalysis
    ) -> float:
        """Apply graph-based adjustments to model probability.
        
        This can help reduce false positives by validating that
        predicted vulnerabilities have corresponding risky code patterns.
        """
        if not graph_analysis.analysis_success:
            return model_prob
        
        adjusted = model_prob
        
        # For borderline positive predictions (0.5-0.7), check if evidence supports it
        if self.threshold <= model_prob < 0.7:
            if graph_analysis.risk_score < 0.1:
                # No risky patterns found, reduce score slightly
                adjusted = model_prob * 0.85
                logger.debug(f"Reduced score from {model_prob:.4f} to {adjusted:.4f} - no risky patterns")
            elif graph_analysis.has_dangerous_flow:
                # Strong evidence of vulnerability, keep or boost slightly
                adjusted = min(1.0, model_prob * 1.05)
                logger.debug(f"Boosted score from {model_prob:.4f} to {adjusted:.4f} - dangerous flow found")
        
        # For borderline negative predictions (0.4-threshold), check if we should elevate
        elif 0.4 <= model_prob < self.threshold:
            if graph_analysis.has_dangerous_flow or graph_analysis.unguarded_dangerous_calls > 0:
                # Risky patterns found, elevate score
                adjusted = model_prob * 1.15
                logger.debug(f"Elevated score from {model_prob:.4f} to {adjusted:.4f} - risky patterns found")
        
        # For very low scores, if there are unguarded dangerous calls, elevate
        elif model_prob < 0.3 and graph_analysis.unguarded_dangerous_calls > 1:
            adjusted = model_prob * 1.2
            logger.debug(f"Elevated low score from {model_prob:.4f} to {adjusted:.4f} - unguarded calls")
        
        return max(0.0, min(1.0, adjusted))
    
    @torch.inference_mode()
    def predict_with_localization(
        self, 
        code: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
        localize_margin: float = 0.05,
    ) -> LocalizationResult:
        """Predict vulnerability using ensemble and localize suspicious code lines.
        
        If graph-based post-processing is enabled, it adjusts scores
        for borderline predictions based on code structure analysis.
        """
        inputs, tokens, token_lines, global_token_indices, slice_pos_mapping = self._preprocess_with_mapping(code)
        
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)
        
        probs = []
        all_attentions = []
        for model in self.models:
            logits, attention = model.forward_with_attention(**inputs)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            probs.append(prob)
            all_attentions.append(attention)
        
        avg_prob = sum(probs) / len(probs)
        
        # Apply isotonic calibration
        calibrated_prob = self._calibrate_probability(avg_prob)
        
        # Apply graph-based post-processing for borderline predictions
        adjusted_prob = calibrated_prob
        if self.use_graph_postprocessing and self.graph_analyzer:
            try:
                graph_analysis = self.graph_analyzer.analyze(code)
                adjusted_prob = self._apply_graph_postprocessing(calibrated_prob, graph_analysis)
            except Exception as e:
                logger.debug(f"Graph post-processing failed: {e}")
        
        # Apply static analysis pattern detection boost
        boost_score, detected_patterns = detect_critical_patterns(code)
        if boost_score > 0:
            adjusted_prob = min(1.0, adjusted_prob + boost_score)
            logger.debug(f"Pattern boost: +{boost_score:.2f}, patterns: {detected_patterns}")
        
        vulnerable = adjusted_prob >= self.threshold
        
        prediction = PredictionResponse(
            vulnerable=vulnerable,
            score=round(adjusted_prob, 4),
            threshold=self.threshold,
            confidence=self._get_confidence(adjusted_prob),
            detected_patterns=detected_patterns,
        )
        
        highlights = []
        if vulnerable and adjusted_prob >= self.threshold + localize_margin:
            avg_global_alpha = sum(a.global_alpha for a in all_attentions) / len(all_attentions)
            avg_slice_token_alpha = sum(a.slice_token_alpha for a in all_attentions) / len(all_attentions)
            avg_slice_level_alpha = sum(a.slice_level_alpha for a in all_attentions) / len(all_attentions)
            avg_slice_seq_alpha = sum(a.slice_seq_alpha for a in all_attentions) / len(all_attentions)
            
            avg_attention = AttentionWeights(
                global_alpha=avg_global_alpha,
                slice_token_alpha=avg_slice_token_alpha,
                slice_level_alpha=avg_slice_level_alpha,
                slice_seq_alpha=avg_slice_seq_alpha,
            )
            
            importance = self._compute_token_importance(
                avg_attention, global_token_indices, slice_pos_mapping, len(tokens)
            )
            
            all_locations = self._aggregate_to_lines(importance, token_lines, tokens, code)
            
            highlights = [
                loc for loc in all_locations 
                if loc.normalized_score >= score_threshold
            ][:top_k]
            
            if not highlights and all_locations:
                highlights = [all_locations[0]]
        
        return LocalizationResult(prediction=prediction, highlights=highlights)


_model_wrapper: Optional[ModelWrapper] = None

def get_model_wrapper() -> ModelWrapper:
    global _model_wrapper
    if _model_wrapper is None:
        _model_wrapper = ModelWrapper()
    return _model_wrapper
