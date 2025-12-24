"""Code Slicing module for vulnerability detection.

Provides backward/forward slicing based on DFG and CFG,
with window-based fallback when parsing fails.

Improvements:
- Forward control dependencies now use post-dominator analysis
  instead of simple BFS reachability for more precise slices
- Configurable use_post_dominator flag to enable/disable
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Any
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import re

import sys
sys.path.insert(0, '/media/hdi/Hdii/Work/C Vul Devign/devign_pipeline')
from src.ast.parser import ParseResult, CFamilyParser
from src.graphs.cfg import CFGBuilder, CFG, BasicBlock
from src.graphs.dfg import DFGBuilder, DFG


class PostDominatorTree:
    """Compute post-dominator tree for control dependency analysis.
    
    A node Y post-dominates node X if every path from X to EXIT goes through Y.
    
    Control dependency: Y is control-dependent on X if:
    1. X has multiple successors (is a branch)
    2. Y is on one path from X but not all paths
    3. Y does not post-dominate X
    
    This gives more precise forward control dependencies than BFS reachability.
    """
    
    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.post_dom: Dict[int, Optional[int]] = {}  # block_id -> immediate post-dominator
        self.post_dom_set: Dict[int, Set[int]] = {}   # block_id -> all post-dominators
        self._compute_post_dominators()
    
    def _compute_post_dominators(self) -> None:
        """Compute post-dominators using iterative algorithm."""
        if not self.cfg or not self.cfg.blocks:
            return
        
        # Get all block IDs
        all_blocks = {b.id for b in self.cfg.blocks}
        
        # Find exit blocks
        exit_ids = set(self.cfg.exit_block_ids) if self.cfg.exit_block_ids else set()
        if not exit_ids:
            # Fallback: blocks with no successors
            exit_ids = {b.id for b in self.cfg.blocks 
                       if not self.cfg.get_successors(b.id)}
        
        # Initialize: exit nodes post-dominate themselves, others = all blocks
        for block_id in all_blocks:
            if block_id in exit_ids:
                self.post_dom_set[block_id] = {block_id}
            else:
                self.post_dom_set[block_id] = all_blocks.copy()
        
        # Iterative refinement (reverse order for post-dominators)
        changed = True
        iterations = 0
        max_iterations = len(all_blocks) * 2 + 10
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for block in self.cfg.blocks:
                if block.id in exit_ids:
                    continue
                
                successors = self.cfg.get_successors(block.id)
                if not successors:
                    continue
                
                # Post-dom(n) = {n} ∪ ∩{Post-dom(s) for s in successors}
                new_post_dom = all_blocks.copy()
                for succ_id in successors:
                    if succ_id in self.post_dom_set:
                        new_post_dom &= self.post_dom_set[succ_id]
                
                new_post_dom.add(block.id)
                
                if new_post_dom != self.post_dom_set[block.id]:
                    self.post_dom_set[block.id] = new_post_dom
                    changed = True
        
        # Compute immediate post-dominators
        for block_id in all_blocks:
            pdoms = self.post_dom_set.get(block_id, set()) - {block_id}
            if not pdoms:
                self.post_dom[block_id] = None
            else:
                # Immediate post-dominator is the one closest to block_id
                # (the one that doesn't post-dominate any other post-dominator)
                ipdom = None
                for candidate in pdoms:
                    is_immediate = True
                    for other in pdoms:
                        if other != candidate:
                            other_pdoms = self.post_dom_set.get(other, set())
                            if candidate in other_pdoms:
                                is_immediate = False
                                break
                    if is_immediate:
                        ipdom = candidate
                        break
                self.post_dom[block_id] = ipdom
    
    def get_control_dependents(self, block_id: int) -> Set[int]:
        """Get all blocks that are control-dependent on the given block.
        
        A block Y is control-dependent on X if:
        - X is a branch (has 2+ successors)
        - There exists a path from X to Y through one successor
        - Y does not post-dominate X
        """
        dependents: Set[int] = set()
        
        if not self.cfg or block_id not in self.post_dom_set:
            return dependents
        
        block = self.cfg.get_block_by_id(block_id)
        if block is None:
            return dependents
        
        successors = self.cfg.get_successors(block_id)
        
        # Only branch nodes create control dependencies
        if len(successors) < 2:
            return dependents
        
        # For each successor path, find nodes that are on that path
        # but don't post-dominate the branch node
        x_post_doms = self.post_dom_set.get(block_id, set())
        
        for succ_id in successors:
            # BFS from this successor to find reachable nodes
            visited: Set[int] = set()
            queue = [succ_id]
            
            while queue:
                curr_id = queue.pop(0)
                if curr_id in visited:
                    continue
                visited.add(curr_id)
                
                # Check if this node is control-dependent on X
                # It is if it doesn't post-dominate X
                if curr_id not in x_post_doms:
                    dependents.add(curr_id)
                
                # Stop expanding if we hit the immediate post-dominator
                # (it marks the merge point)
                if curr_id == self.post_dom.get(block_id):
                    continue
                
                # Continue to successors
                for next_id in self.cfg.get_successors(curr_id):
                    if next_id not in visited:
                        queue.append(next_id)
        
        return dependents
    
    def post_dominates(self, a: int, b: int) -> bool:
        """Check if block A post-dominates block B."""
        return a in self.post_dom_set.get(b, set())


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
    
    # New: Use post-dominator for precise forward control dependencies
    # If True: Only include statements truly control-dependent on criterion
    # If False: Use BFS reachability (old behavior, includes more lines)
    use_post_dominator: bool = True
    
    # Cache post-dominator tree (can be expensive to compute)
    cache_post_dom: bool = True


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
    line_order: List[int] = field(default_factory=list)  # Sorted list of included line numbers
    
    def __post_init__(self):
        if isinstance(self.included_lines, list):
            self.included_lines = set(self.included_lines)


class CodeSlicer:
    """Main code slicing class
    
    Improvements in this version:
    - Forward control dependencies use post-dominator analysis for precision
    - Cached post-dominator tree to avoid recomputation
    """
    
    # Control structure types that affect control flow
    CONTROL_TYPES = {
        'if_statement', 'while_statement', 'for_statement',
        'do_statement', 'switch_statement', 'case_statement',
        'conditional_expression', 'else_clause', 'condition', 'loop_header'
    }
    
    def __init__(self, config: SliceConfig = None):
        self.config = config or SliceConfig()
        self.parser = CFamilyParser()
        
        # Cache for post-dominator trees (CFG id -> PostDominatorTree)
        self._post_dom_cache: Dict[int, PostDominatorTree] = {}
    
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
    
    def forward_window_slice(self, code: str, criterion_lines: List[int], 
                              forward_window: int = None) -> CodeSlice:
        """Window-based forward slice: criterion + next k lines.
        
        Used when graph-based forward slice is empty or too small.
        Only includes lines AFTER criterion (forward direction).
        
        Args:
            code: Source code
            criterion_lines: Starting lines
            forward_window: Number of lines after criterion (default: window_size)
        """
        lines = code.split('\n')
        max_line = len(lines)
        
        if forward_window is None:
            forward_window = self.config.window_size
        
        included_lines: Set[int] = set()
        
        for crit_line in criterion_lines:
            # Include criterion line and forward lines only
            start = crit_line
            end = min(max_line, crit_line + forward_window)
            included_lines.update(range(start, end + 1))
        
        return self._build_slice(code, included_lines, criterion_lines, SliceType.WINDOW)
    
    def slice_pair(self, code: str, criterion_lines: List[int],
                   cfg: Optional[CFG] = None, dfg: Optional[DFG] = None,
                   min_forward_lines: int = 5) -> Tuple[CodeSlice, CodeSlice]:
        """Generate both backward and forward slices, ensuring forward is not empty.
        
        This addresses the issue where 25% of samples have only 1 slice.
        If forward slice is too small, falls back to window-based forward.
        
        Args:
            code: Source code
            criterion_lines: Slicing criterion
            cfg: Optional CFG
            dfg: Optional DFG
            min_forward_lines: Minimum lines for forward slice before fallback
        
        Returns:
            Tuple of (backward_slice, forward_slice)
        """
        # Get backward slice
        original_type = self.config.slice_type
        self.config.slice_type = SliceType.BACKWARD
        backward = self.slice(code, criterion_lines, cfg, dfg)
        
        # Get forward slice
        self.config.slice_type = SliceType.FORWARD
        forward = self.slice(code, criterion_lines, cfg, dfg)
        
        # Restore original config
        self.config.slice_type = original_type
        
        # Check if forward slice is too small
        forward_lines = len(forward.included_lines) - len(set(criterion_lines))
        
        if forward_lines < min_forward_lines or not forward.code.strip():
            # Fallback to window-based forward
            forward = self.forward_window_slice(code, criterion_lines)
        
        return backward, forward
    
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
        """Get lines controlled by target lines (forward).
        
        If use_post_dominator is True (default):
            Uses post-dominator analysis for precise control dependencies.
            Only includes lines that are truly control-dependent on the criterion.
        
        If use_post_dominator is False:
            Uses BFS reachability (old behavior, may include too many lines).
        """
        control_lines: Set[int] = set()
        
        if cfg is None or not cfg.blocks:
            return control_lines
        
        target_block_ids = cfg.get_blocks_for_lines(target_lines)
        
        if self.config.use_post_dominator:
            return self._get_forward_control_deps_postdom(cfg, target_block_ids)
        else:
            return self._get_forward_control_deps_bfs(cfg, target_block_ids)
    
    def _get_forward_control_deps_postdom(self, cfg: CFG, target_block_ids: List[int]) -> Set[int]:
        """Get forward control dependencies using post-dominator analysis.
        
        More precise than BFS: only includes blocks truly control-dependent on criterion.
        """
        control_lines: Set[int] = set()
        
        # Get or create post-dominator tree
        cfg_id = id(cfg)
        if self.config.cache_post_dom and cfg_id in self._post_dom_cache:
            post_dom = self._post_dom_cache[cfg_id]
        else:
            post_dom = PostDominatorTree(cfg)
            if self.config.cache_post_dom:
                self._post_dom_cache[cfg_id] = post_dom
        
        dependent_blocks: Set[int] = set()
        
        for block_id in target_block_ids:
            block = cfg.get_block_by_id(block_id)
            if block is None:
                continue
            
            # Add lines from the target block itself
            for line in range(block.start_line, block.end_line + 1):
                control_lines.add(line)
            
            # If branch node, get control-dependent blocks
            if block.block_type in self.CONTROL_TYPES or len(cfg.get_successors(block_id)) >= 2:
                deps = post_dom.get_control_dependents(block_id)
                dependent_blocks.update(deps)
        
        for dep_block_id in dependent_blocks:
            dep_block = cfg.get_block_by_id(dep_block_id)
            if dep_block is not None:
                for line in range(dep_block.start_line, dep_block.end_line + 1):
                    control_lines.add(line)
        
        return control_lines
    
    def _get_forward_control_deps_bfs(self, cfg: CFG, target_block_ids: List[int]) -> Set[int]:
        """Get forward control dependencies using BFS reachability (old method).
        
        WARNING: May include too many lines. Prefer use_post_dominator=True.
        """
        control_lines: Set[int] = set()
        visited: Set[int] = set()
        queue = list(target_block_ids)
        
        while queue:
            block_id = queue.pop(0)
            if block_id in visited:
                continue
            visited.add(block_id)
            
            block = cfg.get_block_by_id(block_id)
            if block is None:
                continue
            
            for line in range(block.start_line, block.end_line + 1):
                control_lines.add(line)
            
            for succ_id in cfg.get_successors(block_id):
                if succ_id not in visited:
                    queue.append(succ_id)
        
        return control_lines
    
    def clear_cache(self) -> None:
        """Clear the post-dominator cache."""
        self._post_dom_cache.clear()
    
    def get_vuln_line_position(self, code_slice: 'CodeSlice') -> int:
        """Get the index of vulnerability line within the slice's line_order."""
        line_order = code_slice.line_order or sorted(code_slice.included_lines)
        criterion_set = set(code_slice.criterion_lines)
        
        for i, line_num in enumerate(line_order):
            if line_num in criterion_set:
                return i
        
        return len(line_order) // 2
    
    def centered_truncate(self, code_slice: 'CodeSlice', max_tokens: int = 256) -> Tuple[str, List[int]]:
        """
        Truncate slice code centered around vulnerability line.
        
        Args:
            code_slice: The CodeSlice to truncate
            max_tokens: Maximum number of tokens (default 256)
        
        Returns:
            Tuple of (truncated_code, token_line_numbers)
        
        Algorithm:
        1. Find which line index in line_order is the criterion line
        2. Tokenize each statement separately
        3. Find token index of criterion line
        4. Keep window of ±(max_tokens/2) tokens around that index
        """
        if not code_slice.statements:
            return "", []
        
        line_order = code_slice.line_order or sorted(code_slice.included_lines)
        if not line_order:
            return "", []
        
        all_tokens: List[str] = []
        token_line_nums: List[int] = []
        
        for i, stmt in enumerate(code_slice.statements):
            if i < len(line_order):
                line_num = line_order[i]
            else:
                line_num = line_order[-1] if line_order else 0
            
            stmt_tokens = stmt.split()
            all_tokens.extend(stmt_tokens)
            token_line_nums.extend([line_num] * len(stmt_tokens))
        
        if not all_tokens:
            return "", []
        
        if len(all_tokens) <= max_tokens:
            return ' '.join(all_tokens), token_line_nums
        
        vuln_line_idx = self.get_vuln_line_position(code_slice)
        
        target_token_idx = 0
        token_count = 0
        for i, stmt in enumerate(code_slice.statements):
            stmt_token_count = len(stmt.split())
            if i >= vuln_line_idx:
                target_token_idx = token_count + stmt_token_count // 2
                break
            token_count += stmt_token_count
        
        half_window = max_tokens // 2
        start_idx = max(0, target_token_idx - half_window)
        end_idx = min(len(all_tokens), start_idx + max_tokens)
        
        if end_idx - start_idx < max_tokens and end_idx == len(all_tokens):
            start_idx = max(0, end_idx - max_tokens)
        
        truncated_tokens = all_tokens[start_idx:end_idx]
        truncated_line_nums = token_line_nums[start_idx:end_idx]
        
        return ' '.join(truncated_tokens), truncated_line_nums
    
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
            statements=statements,
            line_order=sorted(valid_lines)
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
            statements=[],
            line_order=[]
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
