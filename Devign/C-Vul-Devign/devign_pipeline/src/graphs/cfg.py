"""Control Flow Graph (CFG) Builder for C/C++ code"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
import networkx as nx

from ..ast.parser import ParseResult, ASTNode


class EdgeType(Enum):
    SEQUENTIAL = 'sequential'
    TRUE_BRANCH = 'true'
    FALSE_BRANCH = 'false'
    LOOP_BACK = 'loop_back'
    LOOP_EXIT = 'loop_exit'
    SWITCH_CASE = 'case'
    SWITCH_DEFAULT = 'default'
    GOTO = 'goto'
    RETURN = 'return'
    BREAK = 'break'
    CONTINUE = 'continue'


@dataclass
class BasicBlock:
    id: int
    start_line: int
    end_line: int
    statement_indices: List[int] = field(default_factory=list)
    block_type: str = 'normal'  # 'entry', 'exit', 'normal', 'condition', 'loop_header'
    label: Optional[str] = None

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, BasicBlock):
            return self.id == other.id
        return False


@dataclass
class CFG:
    blocks: List[BasicBlock]
    edges: List[Tuple[int, int, EdgeType]]
    entry_block_id: int
    exit_block_ids: List[int]

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph"""
        G = nx.DiGraph()

        for block in self.blocks:
            G.add_node(
                block.id,
                start_line=block.start_line,
                end_line=block.end_line,
                block_type=block.block_type,
                label=block.label,
                statement_indices=block.statement_indices,
            )

        for src, dst, edge_type in self.edges:
            G.add_edge(src, dst, edge_type=edge_type.value)

        return G

    def get_predecessors(self, block_id: int) -> List[int]:
        """Get predecessor block IDs"""
        return [src for src, dst, _ in self.edges if dst == block_id]

    def get_successors(self, block_id: int) -> List[int]:
        """Get successor block IDs"""
        return [dst for src, dst, _ in self.edges if src == block_id]

    def get_blocks_for_lines(self, lines: List[int]) -> List[int]:
        """Get block IDs that contain given lines"""
        result = []
        lines_set = set(lines)
        for block in self.blocks:
            block_lines = set(range(block.start_line, block.end_line + 1))
            if block_lines & lines_set:
                result.append(block.id)
        return result

    def get_block_by_id(self, block_id: int) -> Optional[BasicBlock]:
        """Get block by ID"""
        for block in self.blocks:
            if block.id == block_id:
                return block
        return None


# Control flow node types in tree-sitter C/C++
CONTROL_FLOW_TYPES = {
    'if_statement',
    'while_statement',
    'for_statement',
    'do_statement',
    'switch_statement',
    'case_statement',
    'goto_statement',
    'break_statement',
    'continue_statement',
    'return_statement',
    'labeled_statement',
}

LOOP_TYPES = {'while_statement', 'for_statement', 'do_statement'}

JUMP_TYPES = {'break_statement', 'continue_statement', 'return_statement', 'goto_statement'}


class CFGBuilder:
    """Build Control Flow Graph from parsed AST"""

    def __init__(self):
        self._reset()

    def _reset(self):
        """Reset builder state"""
        self.blocks: List[BasicBlock] = []
        self.edges: List[Tuple[int, int, EdgeType]] = []
        self.current_block_id = 0
        self.label_to_block: Dict[str, int] = {}
        self.pending_gotos: List[Tuple[int, str]] = []
        self.loop_stack: List[Tuple[int, int]] = []  # (header_block_id, exit_block_id)
        self.switch_stack: List[int] = []  # exit block ids for switches

    def build(self, parse_result: ParseResult) -> Optional[CFG]:
        """Build CFG from ParseResult. Returns None if building fails."""
        if parse_result is None or not parse_result.nodes:
            return None

        self._reset()

        try:
            control_nodes = self._find_control_structures(parse_result)
            self._create_basic_blocks(parse_result, control_nodes)

            if not self.blocks:
                entry = self._new_block(1, 1, 'entry')
                exit_block = self._new_block(1, 1, 'exit')
                self.edges.append((entry.id, exit_block.id, EdgeType.SEQUENTIAL))
                return CFG(
                    blocks=self.blocks,
                    edges=self.edges,
                    entry_block_id=entry.id,
                    exit_block_ids=[exit_block.id],
                )

            self._connect_blocks(parse_result)
            self._resolve_gotos()

            entry_id = self.blocks[0].id if self.blocks else 0
            exit_ids = [b.id for b in self.blocks if b.block_type == 'exit']
            if not exit_ids:
                exit_ids = [self.blocks[-1].id] if self.blocks else []

            return CFG(
                blocks=self.blocks,
                edges=self.edges,
                entry_block_id=entry_id,
                exit_block_ids=exit_ids,
            )

        except Exception:
            return None

    def _new_block(
        self,
        start_line: int,
        end_line: int,
        block_type: str = 'normal',
        label: Optional[str] = None,
    ) -> BasicBlock:
        """Create a new basic block"""
        block = BasicBlock(
            id=self.current_block_id,
            start_line=start_line,
            end_line=end_line,
            statement_indices=[],
            block_type=block_type,
            label=label,
        )
        self.blocks.append(block)
        self.current_block_id += 1
        return block

    def _add_edge(self, src: int, dst: int, edge_type: EdgeType):
        """Add edge if not duplicate"""
        edge = (src, dst, edge_type)
        if edge not in self.edges:
            self.edges.append(edge)

    def _find_control_structures(self, parse_result: ParseResult) -> List[Tuple[int, ASTNode]]:
        """Find control flow statements and their indices"""
        control_nodes = []
        for idx, node in enumerate(parse_result.nodes):
            if node.node_type in CONTROL_FLOW_TYPES:
                control_nodes.append((idx, node))
        return control_nodes

    def _get_function_body(self, parse_result: ParseResult) -> Optional[Tuple[int, ASTNode]]:
        """Find the function body (compound_statement) in the AST"""
        for idx, node in enumerate(parse_result.nodes):
            if node.node_type == 'function_definition':
                for child_idx in node.children_indices:
                    child = parse_result.nodes[child_idx]
                    if child.node_type == 'compound_statement':
                        return (child_idx, child)
        return None

    def _create_basic_blocks(
        self, parse_result: ParseResult, control_nodes: List[Tuple[int, ASTNode]]
    ):
        """Create basic blocks based on control flow boundaries"""
        if not parse_result.nodes:
            return

        func_body = self._get_function_body(parse_result)
        if func_body is None:
            root = parse_result.nodes[parse_result.root_index]
            start_line = root.start_line
            end_line = root.end_line
        else:
            _, body_node = func_body
            start_line = body_node.start_line
            end_line = body_node.end_line

        entry_block = self._new_block(start_line, start_line, 'entry')

        boundaries: Set[int] = {start_line, end_line + 1}

        for idx, node in control_nodes:
            boundaries.add(node.start_line)
            boundaries.add(node.end_line + 1)

            if node.node_type == 'if_statement':
                self._add_if_boundaries(parse_result, idx, node, boundaries)
            elif node.node_type in LOOP_TYPES:
                self._add_loop_boundaries(parse_result, idx, node, boundaries)
            elif node.node_type == 'switch_statement':
                self._add_switch_boundaries(parse_result, idx, node, boundaries)

        sorted_boundaries = sorted(boundaries)

        for i in range(len(sorted_boundaries) - 1):
            block_start = sorted_boundaries[i]
            block_end = sorted_boundaries[i + 1] - 1

            if block_start > block_end:
                continue
            if block_start < start_line or block_end > end_line:
                continue

            block_type = self._determine_block_type(parse_result, block_start, control_nodes)
            label = self._find_label_at_line(parse_result, block_start)

            block = self._new_block(block_start, block_end, block_type, label)

            if label:
                self.label_to_block[label] = block.id

            for idx, node in enumerate(parse_result.nodes):
                if node.start_line >= block_start and node.end_line <= block_end:
                    if node.node_type not in ('compound_statement', 'translation_unit'):
                        block.statement_indices.append(idx)

        exit_block = self._new_block(end_line, end_line, 'exit')

    def _add_if_boundaries(
        self,
        parse_result: ParseResult,
        idx: int,
        node: ASTNode,
        boundaries: Set[int],
    ):
        """Add boundaries for if statement"""
        for child_idx in node.children_indices:
            child = parse_result.nodes[child_idx]
            if child.node_type in ('compound_statement', 'else_clause'):
                boundaries.add(child.start_line)
                boundaries.add(child.end_line + 1)

    def _add_loop_boundaries(
        self,
        parse_result: ParseResult,
        idx: int,
        node: ASTNode,
        boundaries: Set[int],
    ):
        """Add boundaries for loop statement"""
        for child_idx in node.children_indices:
            child = parse_result.nodes[child_idx]
            if child.node_type == 'compound_statement':
                boundaries.add(child.start_line)
                boundaries.add(child.end_line + 1)

    def _add_switch_boundaries(
        self,
        parse_result: ParseResult,
        idx: int,
        node: ASTNode,
        boundaries: Set[int],
    ):
        """Add boundaries for switch statement"""
        for child_idx in node.children_indices:
            child = parse_result.nodes[child_idx]
            if child.node_type == 'compound_statement':
                for case_idx in child.children_indices:
                    case_node = parse_result.nodes[case_idx]
                    if case_node.node_type == 'case_statement':
                        boundaries.add(case_node.start_line)
                        boundaries.add(case_node.end_line + 1)

    def _determine_block_type(
        self,
        parse_result: ParseResult,
        line: int,
        control_nodes: List[Tuple[int, ASTNode]],
    ) -> str:
        """Determine block type based on statements at line"""
        for idx, node in control_nodes:
            if node.start_line == line:
                if node.node_type == 'if_statement':
                    return 'condition'
                if node.node_type in LOOP_TYPES:
                    return 'loop_header'
                if node.node_type == 'switch_statement':
                    return 'condition'
        return 'normal'

    def _find_label_at_line(self, parse_result: ParseResult, line: int) -> Optional[str]:
        """Find goto label at line"""
        for node in parse_result.nodes:
            if node.node_type == 'labeled_statement' and node.start_line == line:
                for child_idx in node.children_indices:
                    child = parse_result.nodes[child_idx]
                    if child.node_type == 'statement_identifier':
                        return child.text
        return None

    def _connect_blocks(self, parse_result: ParseResult):
        """Connect blocks based on control flow"""
        block_by_line: Dict[int, BasicBlock] = {}
        for block in self.blocks:
            for line in range(block.start_line, block.end_line + 1):
                block_by_line[line] = block

        sorted_blocks = sorted(
            [b for b in self.blocks if b.block_type not in ('entry', 'exit')],
            key=lambda b: b.start_line,
        )

        if sorted_blocks and self.blocks:
            entry = next((b for b in self.blocks if b.block_type == 'entry'), None)
            if entry:
                self._add_edge(entry.id, sorted_blocks[0].id, EdgeType.SEQUENTIAL)

        control_map = self._build_control_map(parse_result)

        for i, block in enumerate(sorted_blocks):
            has_control_flow = False

            for stmt_idx in block.statement_indices:
                if stmt_idx >= len(parse_result.nodes):
                    continue
                node = parse_result.nodes[stmt_idx]

                if node.node_type == 'if_statement':
                    self._handle_if_statement(parse_result, stmt_idx, block, block_by_line)
                    has_control_flow = True

                elif node.node_type in LOOP_TYPES:
                    self._handle_loop(parse_result, stmt_idx, node.node_type, block, block_by_line)
                    has_control_flow = True

                elif node.node_type == 'switch_statement':
                    self._handle_switch(parse_result, stmt_idx, block, block_by_line)
                    has_control_flow = True

                elif node.node_type == 'goto_statement':
                    label = self._extract_goto_label(parse_result, stmt_idx)
                    if label:
                        self.pending_gotos.append((block.id, label))
                    has_control_flow = True

                elif node.node_type == 'return_statement':
                    exit_block = next((b for b in self.blocks if b.block_type == 'exit'), None)
                    if exit_block:
                        self._add_edge(block.id, exit_block.id, EdgeType.RETURN)
                    has_control_flow = True

                elif node.node_type == 'break_statement':
                    if self.loop_stack:
                        _, exit_id = self.loop_stack[-1]
                        self._add_edge(block.id, exit_id, EdgeType.BREAK)
                    elif self.switch_stack:
                        self._add_edge(block.id, self.switch_stack[-1], EdgeType.BREAK)
                    has_control_flow = True

                elif node.node_type == 'continue_statement':
                    if self.loop_stack:
                        header_id, _ = self.loop_stack[-1]
                        self._add_edge(block.id, header_id, EdgeType.CONTINUE)
                    has_control_flow = True

            if not has_control_flow:
                if i + 1 < len(sorted_blocks):
                    self._add_edge(block.id, sorted_blocks[i + 1].id, EdgeType.SEQUENTIAL)
                else:
                    exit_block = next((b for b in self.blocks if b.block_type == 'exit'), None)
                    if exit_block:
                        self._add_edge(block.id, exit_block.id, EdgeType.SEQUENTIAL)

    def _build_control_map(self, parse_result: ParseResult) -> Dict[int, ASTNode]:
        """Build map of line -> control node"""
        control_map = {}
        for idx, node in enumerate(parse_result.nodes):
            if node.node_type in CONTROL_FLOW_TYPES:
                control_map[node.start_line] = node
        return control_map

    def _handle_if_statement(
        self,
        parse_result: ParseResult,
        stmt_idx: int,
        block: BasicBlock,
        block_by_line: Dict[int, BasicBlock],
    ):
        """Process if/else structure"""
        node = parse_result.nodes[stmt_idx]

        true_block = None
        false_block = None
        merge_line = node.end_line + 1

        for child_idx in node.children_indices:
            child = parse_result.nodes[child_idx]
            if child.node_type == 'compound_statement':
                if true_block is None:
                    true_block = block_by_line.get(child.start_line)
            elif child.node_type == 'else_clause':
                for else_child_idx in child.children_indices:
                    else_child = parse_result.nodes[else_child_idx]
                    if else_child.node_type in ('compound_statement', 'if_statement'):
                        false_block = block_by_line.get(else_child.start_line)

        if true_block:
            self._add_edge(block.id, true_block.id, EdgeType.TRUE_BRANCH)
        if false_block:
            self._add_edge(block.id, false_block.id, EdgeType.FALSE_BRANCH)
        else:
            merge_block = block_by_line.get(merge_line)
            if merge_block:
                self._add_edge(block.id, merge_block.id, EdgeType.FALSE_BRANCH)

    def _handle_loop(
        self,
        parse_result: ParseResult,
        stmt_idx: int,
        loop_type: str,
        block: BasicBlock,
        block_by_line: Dict[int, BasicBlock],
    ):
        """Process while/for/do-while"""
        node = parse_result.nodes[stmt_idx]

        exit_line = node.end_line + 1
        exit_block = block_by_line.get(exit_line)
        if exit_block is None:
            exit_block = next((b for b in self.blocks if b.block_type == 'exit'), None)

        exit_id = exit_block.id if exit_block else block.id

        self.loop_stack.append((block.id, exit_id))

        body_block = None
        for child_idx in node.children_indices:
            child = parse_result.nodes[child_idx]
            if child.node_type == 'compound_statement':
                body_block = block_by_line.get(child.start_line)
                break

        if loop_type == 'do_statement':
            if body_block:
                self._add_edge(block.id, body_block.id, EdgeType.SEQUENTIAL)
                self._add_edge(body_block.id, block.id, EdgeType.LOOP_BACK)
            if exit_block:
                self._add_edge(block.id, exit_id, EdgeType.LOOP_EXIT)
        else:
            if body_block:
                self._add_edge(block.id, body_block.id, EdgeType.TRUE_BRANCH)
            if exit_block:
                self._add_edge(block.id, exit_id, EdgeType.LOOP_EXIT)

        if self.loop_stack:
            self.loop_stack.pop()

    def _handle_switch(
        self,
        parse_result: ParseResult,
        stmt_idx: int,
        block: BasicBlock,
        block_by_line: Dict[int, BasicBlock],
    ):
        """Process switch/case"""
        node = parse_result.nodes[stmt_idx]

        exit_line = node.end_line + 1
        exit_block = block_by_line.get(exit_line)
        if exit_block is None:
            exit_block = next((b for b in self.blocks if b.block_type == 'exit'), None)

        exit_id = exit_block.id if exit_block else block.id
        self.switch_stack.append(exit_id)

        for child_idx in node.children_indices:
            child = parse_result.nodes[child_idx]
            if child.node_type == 'compound_statement':
                for case_idx in child.children_indices:
                    case_node = parse_result.nodes[case_idx]
                    if case_node.node_type == 'case_statement':
                        case_block = block_by_line.get(case_node.start_line)
                        if case_block:
                            is_default = 'default' in case_node.text.lower()
                            edge_type = EdgeType.SWITCH_DEFAULT if is_default else EdgeType.SWITCH_CASE
                            self._add_edge(block.id, case_block.id, edge_type)

        if self.switch_stack:
            self.switch_stack.pop()

    def _extract_goto_label(self, parse_result: ParseResult, stmt_idx: int) -> Optional[str]:
        """Extract label from goto statement"""
        node = parse_result.nodes[stmt_idx]
        for child_idx in node.children_indices:
            child = parse_result.nodes[child_idx]
            if child.node_type == 'statement_identifier':
                return child.text
        text = node.text
        if 'goto' in text:
            parts = text.replace('goto', '').replace(';', '').strip().split()
            if parts:
                return parts[0]
        return None

    def _resolve_gotos(self):
        """Resolve pending goto edges"""
        for src_block_id, label in self.pending_gotos:
            if label in self.label_to_block:
                dst_block_id = self.label_to_block[label]
                self._add_edge(src_block_id, dst_block_id, EdgeType.GOTO)


def build_cfg(parse_result: ParseResult) -> Optional[CFG]:
    """Convenience function to build CFG"""
    builder = CFGBuilder()
    return builder.build(parse_result)


def serialize_cfg(cfg: CFG) -> dict:
    """Serialize CFG to dict for saving to JSON/pickle"""
    if cfg is None:
        return {}

    return {
        'blocks': [
            {
                'id': b.id,
                'start_line': b.start_line,
                'end_line': b.end_line,
                'statement_indices': b.statement_indices,
                'block_type': b.block_type,
                'label': b.label,
            }
            for b in cfg.blocks
        ],
        'edges': [
            {'src': src, 'dst': dst, 'type': edge_type.value}
            for src, dst, edge_type in cfg.edges
        ],
        'entry_block_id': cfg.entry_block_id,
        'exit_block_ids': cfg.exit_block_ids,
    }


def deserialize_cfg(data: dict) -> Optional[CFG]:
    """Deserialize CFG from dict"""
    if not data or 'blocks' not in data:
        return None

    edge_type_map = {e.value: e for e in EdgeType}

    blocks = [
        BasicBlock(
            id=b['id'],
            start_line=b['start_line'],
            end_line=b['end_line'],
            statement_indices=b.get('statement_indices', []),
            block_type=b.get('block_type', 'normal'),
            label=b.get('label'),
        )
        for b in data['blocks']
    ]

    edges = [
        (e['src'], e['dst'], edge_type_map.get(e['type'], EdgeType.SEQUENTIAL))
        for e in data.get('edges', [])
    ]

    return CFG(
        blocks=blocks,
        edges=edges,
        entry_block_id=data.get('entry_block_id', 0),
        exit_block_ids=data.get('exit_block_ids', []),
    )
