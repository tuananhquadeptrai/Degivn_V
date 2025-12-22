"""Code Slicing module for vulnerability detection.

Provides backward/forward slicing based on DFG and CFG,
with window-based fallback when parsing fails.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Any
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

import sys
sys.path.insert(0, '/media/hdi/Hdii/Work/C Vul Devign/devign_pipeline')
from src.ast.parser import ParseResult, CFamilyParser
from src.graphs.cfg import CFGBuilder, CFG, BasicBlock
from src.graphs.dfg import DFGBuilder, DFG


class SliceType(Enum):
    BACKWARD = 'backward'   # Data/control deps leading to criterion
    FORWARD = 'forward'     # Effects of criterion
    BOTH = 'both'           # Combined
    WINDOW = 'window'       # Simple ±k lines


@dataclass
class SliceConfig:
    slice_type: SliceType = SliceType.BACKWARD
    window_size: int = 15        # For window-based fallback
    include_control_deps: bool = True   # Include if/loop conditions
    include_data_deps: bool = True      # Include def-use chains
    max_depth: int = 5           # Max recursion depth for slicing
    remove_comments: bool = True # Remove comments from sliced code
    normalize_output: bool = True # Normalize whitespace/indentation


@dataclass
class CodeSlice:
    code: str                    # Sliced code as string
    original_code: str           # Original full code
    start_line: int
    end_line: int
    included_lines: Set[int]     # Set of line numbers in slice
    criterion_lines: List[int]   # Original slicing criterion (vul_lines)
    slice_type: SliceType
    statements: List[str]        # Individual statements in slice
    
    def __post_init__(self):
        if isinstance(self.included_lines, list):
            self.included_lines = set(self.included_lines)


class CodeSlicer:
    """Main code slicing class"""
    
    # Control structure types that affect control flow
    CONTROL_TYPES = {
        'if_statement', 'while_statement', 'for_statement',
        'do_statement', 'switch_statement', 'case_statement',
        'conditional_expression', 'else_clause', 'condition', 'loop_header'
    }
    
    def __init__(self, config: SliceConfig = None):
        self.config = config or SliceConfig()
        self.parser = CFamilyParser()
    
    def slice(self, code: str, criterion_lines: List[int], 
              cfg: Optional[CFG] = None, dfg: Optional[DFG] = None) -> CodeSlice:
        """
        Main slicing function.
        Falls back to window-based if AST/graphs fail.
        
        Args:
            code: Source code to slice
            criterion_lines: Lines to use as slicing criterion
            cfg: Optional precomputed CFG (if None, will be built internally)
            dfg: Optional precomputed DFG (if None, will be built internally)
        """
        if not code or not code.strip():
            return self._empty_slice(code, criterion_lines)
        
        # Validate criterion lines
        lines = code.split('\n')
        max_line = len(lines)
        valid_criterion = [l for l in criterion_lines if 1 <= l <= max_line]
        
        if not valid_criterion:
            valid_criterion = criterion_lines[:1] if criterion_lines else [1]
        
        # Remove comments first if configured
        processed_code = code
        if self.config.remove_comments:
            processed_code = self.remove_comments(code)
        
        # Try graph-based slicing
        try:
            # Only parse and build graphs if not provided
            if cfg is None and dfg is None:
                parse_result = self.parser.parse_with_fallback(processed_code)
                if parse_result is None or not parse_result.nodes:
                    return self.window_slice(processed_code, valid_criterion)
                
                # Build CFG
                cfg_builder = CFGBuilder()
                cfg = cfg_builder.build(parse_result)
                
                # Build DFG
                dfg_builder = DFGBuilder()
                dfg = dfg_builder.build(parse_result, focus_lines=valid_criterion)
            
            if cfg is None and dfg is None:
                return self.window_slice(processed_code, valid_criterion)
            
            # Perform slicing based on type
            if self.config.slice_type == SliceType.BACKWARD:
                return self.backward_slice(processed_code, valid_criterion, cfg, dfg)
            elif self.config.slice_type == SliceType.FORWARD:
                return self.forward_slice(processed_code, valid_criterion, cfg, dfg)
            elif self.config.slice_type == SliceType.BOTH:
                backward = self.backward_slice(processed_code, valid_criterion, cfg, dfg)
                forward = self.forward_slice(processed_code, valid_criterion, cfg, dfg)
                combined_lines = backward.included_lines | forward.included_lines
                return self._build_slice(processed_code, combined_lines, 
                                         valid_criterion, SliceType.BOTH)
            else:
                return self.window_slice(processed_code, valid_criterion)
                
        except Exception:
            return self.window_slice(processed_code, valid_criterion)
    
    def backward_slice(self, code: str, criterion_lines: List[int],
                       cfg: Optional[CFG] = None, 
                       dfg: Optional[DFG] = None) -> CodeSlice:
        """
        Backward slicing: find all statements that affect criterion lines.
        Uses DFG for data dependencies, CFG for control dependencies.
        """
        included_lines: Set[int] = set(criterion_lines)
        
        if cfg is None and dfg is None:
            try:
                parse_result = self.parser.parse_with_fallback(code)
                if parse_result and parse_result.nodes:
                    if cfg is None:
                        cfg_builder = CFGBuilder()
                        cfg = cfg_builder.build(parse_result)
                    if dfg is None:
                        dfg_builder = DFGBuilder()
                        dfg = dfg_builder.build(parse_result, focus_lines=criterion_lines)
            except Exception:
                return self.window_slice(code, criterion_lines)
        
        if cfg is None and dfg is None:
            return self.window_slice(code, criterion_lines)
        
        # Get data dependencies
        if self.config.include_data_deps and dfg is not None:
            data_deps = self._get_data_dependencies(dfg, list(included_lines))
            included_lines.update(data_deps)
        
        # Get control dependencies
        if self.config.include_control_deps and cfg is not None:
            control_deps = self._get_control_dependencies(cfg, list(included_lines))
            included_lines.update(control_deps)
        
        # Iteratively expand (up to max_depth)
        for _ in range(self.config.max_depth - 1):
            new_lines = set()
            
            if self.config.include_data_deps and dfg is not None:
                new_lines.update(self._get_data_dependencies(dfg, list(included_lines)))
            
            if self.config.include_control_deps and cfg is not None:
                new_lines.update(self._get_control_dependencies(cfg, list(included_lines)))
            
            if not new_lines - included_lines:
                break
            included_lines.update(new_lines)
        
        return self._build_slice(code, included_lines, criterion_lines, SliceType.BACKWARD)
    
    def forward_slice(self, code: str, criterion_lines: List[int],
                      cfg: Optional[CFG] = None,
                      dfg: Optional[DFG] = None) -> CodeSlice:
        """
        Forward slicing: find all statements affected by criterion lines.
        """
        included_lines: Set[int] = set(criterion_lines)
        
        if cfg is None and dfg is None:
            try:
                parse_result = self.parser.parse_with_fallback(code)
                if parse_result and parse_result.nodes:
                    if cfg is None:
                        cfg_builder = CFGBuilder()
                        cfg = cfg_builder.build(parse_result)
                    if dfg is None:
                        dfg_builder = DFGBuilder()
                        dfg = dfg_builder.build(parse_result, focus_lines=criterion_lines)
            except Exception:
                return self.window_slice(code, criterion_lines)
        
        if cfg is None and dfg is None:
            return self.window_slice(code, criterion_lines)
        
        # Get forward data dependencies (dependents)
        if self.config.include_data_deps and dfg is not None:
            forward_deps = self._get_forward_data_dependencies(dfg, list(included_lines))
            included_lines.update(forward_deps)
        
        # Get forward control flow
        if self.config.include_control_deps and cfg is not None:
            forward_control = self._get_forward_control_dependencies(cfg, list(included_lines))
            included_lines.update(forward_control)
        
        # Iteratively expand
        for _ in range(self.config.max_depth - 1):
            new_lines = set()
            
            if self.config.include_data_deps and dfg is not None:
                new_lines.update(self._get_forward_data_dependencies(dfg, list(included_lines)))
            
            if self.config.include_control_deps and cfg is not None:
                new_lines.update(self._get_forward_control_dependencies(cfg, list(included_lines)))
            
            if not new_lines - included_lines:
                break
            included_lines.update(new_lines)
        
        return self._build_slice(code, included_lines, criterion_lines, SliceType.FORWARD)
    
    def window_slice(self, code: str, criterion_lines: List[int]) -> CodeSlice:
        """
        Simple window-based slicing: ±k lines around criterion.
        Used as fallback or when cfg/dfg not available.
        """
        lines = code.split('\n')
        max_line = len(lines)
        
        included_lines: Set[int] = set()
        
        for crit_line in criterion_lines:
            start = max(1, crit_line - self.config.window_size)
            end = min(max_line, crit_line + self.config.window_size)
            included_lines.update(range(start, end + 1))
        
        return self._build_slice(code, included_lines, criterion_lines, SliceType.WINDOW)
    
    def remove_comments(self, code: str) -> str:
        """
        Remove C/C++ comments using tree-sitter.
        Preserves line numbers (replaces comments with whitespace).
        """
        try:
            ast_root = self.parser.get_root(code)
            if ast_root is None:
                return self.remove_comments_regex(code)
            
            # Find all comment nodes
            comment_ranges: List[Tuple[int, int]] = []
            
            def find_comments(node: Any):
                if node.type in ('comment', 'line_comment', 'block_comment'):
                    comment_ranges.append((node.start_byte, node.end_byte))
                for child in node.children:
                    find_comments(child)
            
            find_comments(ast_root)
            
            if not comment_ranges:
                return code
            
            # Sort by start position (descending) to replace from end
            comment_ranges.sort(key=lambda x: x[0], reverse=True)
            
            code_bytes = code.encode('utf-8')
            result = bytearray(code_bytes)
            
            for start, end in comment_ranges:
                # Replace with spaces, preserving newlines
                for i in range(start, min(end, len(result))):
                    if result[i] != ord('\n'):
                        result[i] = ord(' ')
            
            return result.decode('utf-8')
            
        except Exception:
            return self.remove_comments_regex(code)
    
    def remove_comments_regex(self, code: str) -> str:
        """
        Fallback: Remove comments using regex.
        Less accurate but doesn't need parser.
        """
        # Remove block comments (/* ... */) while preserving newlines
        def replace_block_comment(match):
            comment = match.group(0)
            return re.sub(r'[^\n]', ' ', comment)
        
        result = re.sub(r'/\*.*?\*/', replace_block_comment, code, flags=re.DOTALL)
        
        # Remove line comments (// ...)
        result = re.sub(r'//[^\n]*', lambda m: ' ' * len(m.group(0)), result)
        
        return result
    
    def _get_data_dependencies(self, dfg: DFG, target_lines: List[int]) -> Set[int]:
        """Get lines with data dependencies to target lines (backward)"""
        dep_lines: Set[int] = set()
        
        if dfg is None or not dfg.nodes:
            return dep_lines
        
        # Get all node indices at target lines
        target_node_indices = dfg.get_nodes_for_lines(target_lines)
        visited: Set[int] = set()
        
        # BFS backward through def-use chains
        queue = list(target_node_indices)
        
        while queue:
            node_idx = queue.pop(0)
            if node_idx in visited or node_idx >= len(dfg.nodes):
                continue
            visited.add(node_idx)
            
            node = dfg.nodes[node_idx]
            dep_lines.add(node.line)
            
            # Find edges pointing TO this node (backward)
            for edge in dfg.edges:
                if edge.to_idx == node_idx and edge.from_idx not in visited:
                    queue.append(edge.from_idx)
        
        return dep_lines
    
    def _get_forward_data_dependencies(self, dfg: DFG, target_lines: List[int]) -> Set[int]:
        """Get lines that depend on target lines (forward)"""
        dep_lines: Set[int] = set()
        
        if dfg is None or not dfg.nodes:
            return dep_lines
        
        target_node_indices = dfg.get_nodes_for_lines(target_lines)
        visited: Set[int] = set()
        
        # BFS forward through def-use chains
        queue = list(target_node_indices)
        
        while queue:
            node_idx = queue.pop(0)
            if node_idx in visited or node_idx >= len(dfg.nodes):
                continue
            visited.add(node_idx)
            
            node = dfg.nodes[node_idx]
            dep_lines.add(node.line)
            
            # Find edges FROM this node (forward)
            for edge in dfg.edges:
                if edge.from_idx == node_idx and edge.to_idx not in visited:
                    queue.append(edge.to_idx)
        
        return dep_lines
    
    def _get_control_dependencies(self, cfg: CFG, target_lines: List[int]) -> Set[int]:
        """Get lines with control dependencies (conditions that guard target)"""
        control_lines: Set[int] = set()
        
        if cfg is None or not cfg.blocks:
            return control_lines
        
        # Get all CFG blocks containing target lines
        target_block_ids = cfg.get_blocks_for_lines(target_lines)
        visited: Set[int] = set()
        
        # BFS backwards through CFG
        queue = list(target_block_ids)
        
        while queue:
            block_id = queue.pop(0)
            if block_id in visited:
                continue
            visited.add(block_id)
            
            block = cfg.get_block_by_id(block_id)
            if block is None:
                continue
            
            # Add lines from control blocks
            if block.block_type in self.CONTROL_TYPES:
                for line in range(block.start_line, block.end_line + 1):
                    control_lines.add(line)
            
            # Add predecessors to queue
            for pred_id in cfg.get_predecessors(block_id):
                if pred_id not in visited:
                    queue.append(pred_id)
        
        return control_lines
    
    def _get_forward_control_dependencies(self, cfg: CFG, target_lines: List[int]) -> Set[int]:
        """Get lines controlled by target lines (forward)"""
        control_lines: Set[int] = set()
        
        if cfg is None or not cfg.blocks:
            return control_lines
        
        target_block_ids = cfg.get_blocks_for_lines(target_lines)
        visited: Set[int] = set()
        
        # BFS forward through CFG
        queue = list(target_block_ids)
        
        while queue:
            block_id = queue.pop(0)
            if block_id in visited:
                continue
            visited.add(block_id)
            
            block = cfg.get_block_by_id(block_id)
            if block is None:
                continue
            
            # Add all lines from this block
            for line in range(block.start_line, block.end_line + 1):
                control_lines.add(line)
            
            # Add successors to queue
            for succ_id in cfg.get_successors(block_id):
                if succ_id not in visited:
                    queue.append(succ_id)
        
        return control_lines
    
    def _extract_lines(self, code: str, line_numbers: Set[int]) -> Tuple[str, List[str]]:
        """Extract specific lines from code, return (combined_code, list_of_statements)"""
        lines = code.split('\n')
        statements = []
        
        sorted_lines = sorted(line_numbers)
        
        for line_num in sorted_lines:
            if 1 <= line_num <= len(lines):
                line_content = lines[line_num - 1]
                if line_content.strip():  # Skip empty lines
                    statements.append(line_content)
        
        combined = '\n'.join(statements)
        return combined, statements
    
    def _normalize_slice(self, code: str, lines: Set[int]) -> str:
        """Normalize sliced code: remove empty lines, fix indentation"""
        code_lines = code.split('\n')
        result_lines = []
        
        sorted_lines = sorted(lines)
        
        # Find minimum indentation
        min_indent = float('inf')
        for line_num in sorted_lines:
            if 1 <= line_num <= len(code_lines):
                line = code_lines[line_num - 1]
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
        
        if min_indent == float('inf'):
            min_indent = 0
        
        # Extract and dedent lines
        for line_num in sorted_lines:
            if 1 <= line_num <= len(code_lines):
                line = code_lines[line_num - 1]
                if line.strip():
                    # Remove common indentation
                    if len(line) >= min_indent:
                        result_lines.append(line[int(min_indent):])
                    else:
                        result_lines.append(line.lstrip())
        
        return '\n'.join(result_lines)
    
    def _build_slice(self, code: str, included_lines: Set[int],
                     criterion_lines: List[int], slice_type: SliceType) -> CodeSlice:
        """Build CodeSlice from included lines"""
        if not included_lines:
            return self._empty_slice(code, criterion_lines)
        
        lines = code.split('\n')
        max_line = len(lines)
        
        # Filter valid lines
        valid_lines = {l for l in included_lines if 1 <= l <= max_line}
        
        if not valid_lines:
            return self._empty_slice(code, criterion_lines)
        
        # Extract code and statements
        if self.config.normalize_output:
            sliced_code = self._normalize_slice(code, valid_lines)
        else:
            sliced_code, _ = self._extract_lines(code, valid_lines)
        
        _, statements = self._extract_lines(code, valid_lines)
        
        return CodeSlice(
            code=sliced_code,
            original_code=code,
            start_line=min(valid_lines),
            end_line=max(valid_lines),
            included_lines=valid_lines,
            criterion_lines=criterion_lines,
            slice_type=slice_type,
            statements=statements
        )
    
    def _empty_slice(self, code: str, criterion_lines: List[int]) -> CodeSlice:
        """Return empty slice"""
        return CodeSlice(
            code="",
            original_code=code,
            start_line=0,
            end_line=0,
            included_lines=set(),
            criterion_lines=criterion_lines,
            slice_type=self.config.slice_type,
            statements=[]
        )


def _slice_single(args: Tuple[str, List[int], SliceConfig]) -> CodeSlice:
    """Helper for multiprocessing - slice a single code sample"""
    code, criterion_lines, config = args
    slicer = CodeSlicer(config)
    return slicer.slice(code, criterion_lines)


def slice_batch(codes: List[str], criterion_lines_list: List[List[int]],
                config: SliceConfig = None, n_jobs: int = 4) -> List[CodeSlice]:
    """
    Batch slicing with multiprocessing.
    For efficient processing of entire dataset.
    
    Args:
        codes: List of source code strings
        criterion_lines_list: List of criterion lines for each code
        config: Slicing configuration
        n_jobs: Number of parallel workers
    
    Returns:
        List of CodeSlice objects
    """
    if config is None:
        config = SliceConfig()
    
    if len(codes) != len(criterion_lines_list):
        raise ValueError("codes and criterion_lines_list must have same length")
    
    # Prepare arguments
    args_list = [(code, lines, config) for code, lines in zip(codes, criterion_lines_list)]
    
    # Single-threaded for small batches
    if len(codes) <= 10 or n_jobs == 1:
        return [_slice_single(args) for args in args_list]
    
    # Multiprocessing for large batches
    results: List[Optional[CodeSlice]] = [None] * len(codes)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_idx = {
            executor.submit(_slice_single, args): idx 
            for idx, args in enumerate(args_list)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                # Fallback to window slice on error
                slicer = CodeSlicer(config)
                results[idx] = slicer.window_slice(codes[idx], criterion_lines_list[idx])
    
    return results


if __name__ == "__main__":
    # Test code
    test_code = '''int main() {
    int x = 10;
    int y = 20;
    // This is a comment
    int z = x + y;
    /* Block comment
       spanning multiple lines */
    if (z > 25) {
        printf("Large sum\\n");
    }
    return z;
}'''
    
    config = SliceConfig(
        slice_type=SliceType.BACKWARD,
        window_size=5,
        include_control_deps=True,
        include_data_deps=True
    )
    
    slicer = CodeSlicer(config)
    
    # Test comment removal
    print("=== Comment Removal ===")
    clean_code = slicer.remove_comments(test_code)
    print(clean_code)
    
    # Test backward slice on line with printf
    print("\n=== Backward Slice (criterion: line 9) ===")
    result = slicer.slice(test_code, [9])
    print(f"Included lines: {sorted(result.included_lines)}")
    print(f"Sliced code:\n{result.code}")
    
    # Test window slice
    print("\n=== Window Slice ===")
    config.slice_type = SliceType.WINDOW
    slicer = CodeSlicer(config)
    result = slicer.window_slice(test_code, [6])
    print(f"Included lines: {sorted(result.included_lines)}")
