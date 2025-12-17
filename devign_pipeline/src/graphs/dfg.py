"""Data Flow Graph (DFG) Builder for C/C++ code analysis"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
from collections import defaultdict
import networkx as nx

from ..ast.parser import ParseResult, ASTNode


class UseType(Enum):
    DEF = 'def'
    USE = 'use'
    PARAM = 'param'
    RETURN = 'return'
    CALL_ARG = 'call_arg'
    ADDRESS = 'address'
    DEREF = 'deref'


@dataclass
class VariableAccess:
    var_name: str
    line: int
    col: int
    access_type: UseType
    node_index: int
    context: str = ''


@dataclass
class DFGEdge:
    from_idx: int
    to_idx: int
    edge_type: str


@dataclass
class DFG:
    nodes: List[VariableAccess]
    edges: List[DFGEdge]
    var_defs: Dict[str, List[int]]
    var_uses: Dict[str, List[int]]

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph"""
        G = nx.DiGraph()
        
        for idx, node in enumerate(self.nodes):
            G.add_node(idx, 
                       var_name=node.var_name,
                       line=node.line,
                       col=node.col,
                       access_type=node.access_type.value,
                       context=node.context)
        
        for edge in self.edges:
            G.add_edge(edge.from_idx, edge.to_idx, edge_type=edge.edge_type)
        
        return G

    def get_def_use_chains(self, var_name: str) -> List[Tuple[int, int]]:
        """Get all def-use pairs for a variable"""
        chains = []
        defs = self.var_defs.get(var_name, [])
        uses = self.var_uses.get(var_name, [])
        
        for def_idx in defs:
            def_node = self.nodes[def_idx]
            for use_idx in uses:
                use_node = self.nodes[use_idx]
                if use_node.line >= def_node.line:
                    chains.append((def_idx, use_idx))
        
        return chains

    def get_reaching_definitions(self, use_idx: int) -> List[int]:
        """Get all definitions that can reach a use"""
        reaching = []
        use_node = self.nodes[use_idx]
        var_name = use_node.var_name
        
        defs = self.var_defs.get(var_name, [])
        for def_idx in defs:
            def_node = self.nodes[def_idx]
            if def_node.line <= use_node.line:
                reaching.append(def_idx)
        
        return reaching

    def get_nodes_for_lines(self, lines: List[int]) -> List[int]:
        """Get node indices for given lines"""
        line_set = set(lines)
        return [idx for idx, node in enumerate(self.nodes) if node.line in line_set]


class DFGBuilder:
    """Build Data Flow Graph from parsed C/C++ code"""
    
    DEF_NODE_TYPES = {
        'declaration', 'init_declarator', 'parameter_declaration',
        'assignment_expression', 'update_expression', 'compound_assignment_expr',
    }
    
    USE_NODE_TYPES = {
        'identifier', 'field_expression', 'subscript_expression',
    }
    
    POINTER_OPS = {'pointer_expression', 'address_of_expression'}
    
    def __init__(self, scope_limit: int = 50):
        self.scope_limit = scope_limit
        self._current_parse: Optional[ParseResult] = None

    def build(self, parse_result: ParseResult, 
              focus_lines: Optional[List[int]] = None) -> Optional[DFG]:
        """Build DFG from ParseResult"""
        if not parse_result.nodes:
            return None
        
        self._current_parse = parse_result
        
        if focus_lines:
            min_line = max(1, min(focus_lines) - self.scope_limit)
            max_line = max(focus_lines) + self.scope_limit
        else:
            min_line = 0
            max_line = 99999
        
        return self.build_from_range(parse_result, min_line, max_line)

    def build_from_range(self, parse_result: ParseResult, 
                         start_line: int, end_line: int) -> Optional[DFG]:
        """Build DFG for specific line range"""
        if not parse_result.nodes:
            return None
        
        self._current_parse = parse_result
        
        accesses = self._extract_variable_accesses(parse_result, start_line, end_line)
        call_accesses = self._analyze_function_calls(parse_result, start_line, end_line)
        accesses.extend(call_accesses)
        
        accesses.sort(key=lambda x: (x.line, x.col))
        
        var_defs: Dict[str, List[int]] = defaultdict(list)
        var_uses: Dict[str, List[int]] = defaultdict(list)
        
        for idx, access in enumerate(accesses):
            if access.access_type in (UseType.DEF, UseType.PARAM):
                var_defs[access.var_name].append(idx)
            else:
                var_uses[access.var_name].append(idx)
        
        edges = []
        edges.extend(self._build_def_use_chains(accesses))
        edges.extend(self._analyze_pointer_flow(accesses))
        
        return DFG(
            nodes=accesses,
            edges=edges,
            var_defs=dict(var_defs),
            var_uses=dict(var_uses),
        )

    def _extract_variable_accesses(self, parse_result: ParseResult, 
                                   start_line: int = 0, 
                                   end_line: int = 99999) -> List[VariableAccess]:
        """Extract variable definitions and uses from AST"""
        accesses: List[VariableAccess] = []
        seen: Set[Tuple[str, int, int, UseType]] = set()
        
        for idx, node in enumerate(parse_result.nodes):
            if not (start_line <= node.start_line <= end_line):
                continue
            
            if node.node_type == 'parameter_declaration':
                var_name = self._extract_param_name(parse_result, idx)
                if var_name:
                    key = (var_name, node.start_line, node.start_col, UseType.PARAM)
                    if key not in seen:
                        seen.add(key)
                        accesses.append(VariableAccess(
                            var_name=var_name,
                            line=node.start_line,
                            col=node.start_col,
                            access_type=UseType.PARAM,
                            node_index=idx,
                            context='parameter',
                        ))
            
            elif node.node_type in ('declaration', 'init_declarator'):
                var_names = self._extract_declared_vars(parse_result, idx)
                for var_name in var_names:
                    key = (var_name, node.start_line, node.start_col, UseType.DEF)
                    if key not in seen:
                        seen.add(key)
                        accesses.append(VariableAccess(
                            var_name=var_name,
                            line=node.start_line,
                            col=node.start_col,
                            access_type=UseType.DEF,
                            node_index=idx,
                            context='declaration',
                        ))
            
            elif node.node_type == 'assignment_expression':
                lhs_var = self._extract_lhs_var(parse_result, idx)
                if lhs_var:
                    key = (lhs_var, node.start_line, node.start_col, UseType.DEF)
                    if key not in seen:
                        seen.add(key)
                        accesses.append(VariableAccess(
                            var_name=lhs_var,
                            line=node.start_line,
                            col=node.start_col,
                            access_type=UseType.DEF,
                            node_index=idx,
                            context='assignment',
                        ))
                
                rhs_vars = self._extract_rhs_vars(parse_result, idx)
                for var_name in rhs_vars:
                    key = (var_name, node.start_line, node.start_col + 1, UseType.USE)
                    if key not in seen:
                        seen.add(key)
                        accesses.append(VariableAccess(
                            var_name=var_name,
                            line=node.start_line,
                            col=node.start_col + 1,
                            access_type=UseType.USE,
                            node_index=idx,
                            context='rhs',
                        ))
            
            elif node.node_type == 'update_expression':
                var_name = self._extract_update_var(parse_result, idx)
                if var_name:
                    key = (var_name, node.start_line, node.start_col, UseType.DEF)
                    if key not in seen:
                        seen.add(key)
                        accesses.append(VariableAccess(
                            var_name=var_name,
                            line=node.start_line,
                            col=node.start_col,
                            access_type=UseType.DEF,
                            node_index=idx,
                            context='update',
                        ))
            
            elif node.node_type == 'return_statement':
                ret_vars = self._extract_return_vars(parse_result, idx)
                for var_name in ret_vars:
                    key = (var_name, node.start_line, node.start_col, UseType.RETURN)
                    if key not in seen:
                        seen.add(key)
                        accesses.append(VariableAccess(
                            var_name=var_name,
                            line=node.start_line,
                            col=node.start_col,
                            access_type=UseType.RETURN,
                            node_index=idx,
                            context='return',
                        ))
            
            elif node.node_type == 'pointer_expression':
                ptr_info = self._extract_pointer_access(parse_result, idx)
                if ptr_info:
                    var_name, access_type = ptr_info
                    key = (var_name, node.start_line, node.start_col, access_type)
                    if key not in seen:
                        seen.add(key)
                        accesses.append(VariableAccess(
                            var_name=var_name,
                            line=node.start_line,
                            col=node.start_col,
                            access_type=access_type,
                            node_index=idx,
                            context='pointer',
                        ))
            
            elif node.node_type == 'identifier':
                parent_idx = node.parent_index
                if parent_idx is not None:
                    parent = parse_result.nodes[parent_idx]
                    if parent.node_type not in self.DEF_NODE_TYPES and \
                       parent.node_type not in ('parameter_declaration', 'function_declarator', 
                                                 'type_identifier', 'field_identifier'):
                        var_name = node.text.strip()
                        if var_name and not var_name.isupper():
                            key = (var_name, node.start_line, node.start_col, UseType.USE)
                            if key not in seen:
                                seen.add(key)
                                accesses.append(VariableAccess(
                                    var_name=var_name,
                                    line=node.start_line,
                                    col=node.start_col,
                                    access_type=UseType.USE,
                                    node_index=idx,
                                    context='expression',
                                ))
        
        return accesses

    def _build_def_use_chains(self, accesses: List[VariableAccess]) -> List[DFGEdge]:
        """Build edges from definitions to uses"""
        edges: List[DFGEdge] = []
        
        var_defs: Dict[str, List[Tuple[int, VariableAccess]]] = defaultdict(list)
        
        for idx, access in enumerate(accesses):
            if access.access_type in (UseType.DEF, UseType.PARAM):
                var_defs[access.var_name].append((idx, access))
        
        for idx, access in enumerate(accesses):
            if access.access_type in (UseType.USE, UseType.CALL_ARG, UseType.RETURN, 
                                       UseType.ADDRESS, UseType.DEREF):
                defs = var_defs.get(access.var_name, [])
                
                reaching_def = None
                for def_idx, def_access in reversed(defs):
                    if def_access.line < access.line or \
                       (def_access.line == access.line and def_access.col < access.col):
                        reaching_def = def_idx
                        break
                
                if reaching_def is not None:
                    edges.append(DFGEdge(
                        from_idx=reaching_def,
                        to_idx=idx,
                        edge_type='def-use',
                    ))
        
        return edges

    def _analyze_pointer_flow(self, accesses: List[VariableAccess]) -> List[DFGEdge]:
        """Track pointer assignments and dereferences"""
        edges: List[DFGEdge] = []
        
        addr_ops: List[Tuple[int, VariableAccess]] = []
        deref_ops: List[Tuple[int, VariableAccess]] = []
        
        for idx, access in enumerate(accesses):
            if access.access_type == UseType.ADDRESS:
                addr_ops.append((idx, access))
            elif access.access_type == UseType.DEREF:
                deref_ops.append((idx, access))
        
        for addr_idx, addr_access in addr_ops:
            for deref_idx, deref_access in deref_ops:
                if deref_access.line > addr_access.line:
                    edges.append(DFGEdge(
                        from_idx=addr_idx,
                        to_idx=deref_idx,
                        edge_type='ptr-flow',
                    ))
        
        return edges

    def _analyze_function_calls(self, parse_result: ParseResult, 
                                start_line: int, end_line: int) -> List[VariableAccess]:
        """Extract function call arguments as uses"""
        accesses: List[VariableAccess] = []
        seen: Set[Tuple[str, int, int]] = set()
        
        for idx, node in enumerate(parse_result.nodes):
            if not (start_line <= node.start_line <= end_line):
                continue
            
            if node.node_type == 'call_expression':
                func_name = self._extract_func_name(parse_result, idx)
                args = self._extract_call_args(parse_result, idx)
                
                for arg_name in args:
                    key = (arg_name, node.start_line, node.start_col)
                    if key not in seen:
                        seen.add(key)
                        accesses.append(VariableAccess(
                            var_name=arg_name,
                            line=node.start_line,
                            col=node.start_col,
                            access_type=UseType.CALL_ARG,
                            node_index=idx,
                            context=f'call:{func_name}',
                        ))
        
        return accesses

    def _resolve_reaching_defs(self, uses: List[VariableAccess], 
                               defs: Dict[str, List[VariableAccess]]) -> List[DFGEdge]:
        """For each use, find reaching definitions"""
        edges: List[DFGEdge] = []
        return edges

    def _extract_param_name(self, parse_result: ParseResult, node_idx: int) -> Optional[str]:
        """Extract parameter name from parameter_declaration"""
        node = parse_result.nodes[node_idx]
        for child_idx in node.children_indices:
            child = parse_result.nodes[child_idx]
            if child.node_type == 'identifier':
                return child.text.strip()
            if child.node_type == 'pointer_declarator':
                return self._find_identifier_in_subtree(parse_result, child_idx)
        return None

    def _extract_declared_vars(self, parse_result: ParseResult, node_idx: int) -> List[str]:
        """Extract variable names from declaration"""
        vars_found: List[str] = []
        node = parse_result.nodes[node_idx]
        
        for child_idx in node.children_indices:
            child = parse_result.nodes[child_idx]
            if child.node_type == 'init_declarator':
                var_name = self._find_identifier_in_subtree(parse_result, child_idx)
                if var_name:
                    vars_found.append(var_name)
            elif child.node_type == 'identifier':
                vars_found.append(child.text.strip())
            elif child.node_type in ('pointer_declarator', 'array_declarator'):
                var_name = self._find_identifier_in_subtree(parse_result, child_idx)
                if var_name:
                    vars_found.append(var_name)
        
        return vars_found

    def _extract_lhs_var(self, parse_result: ParseResult, node_idx: int) -> Optional[str]:
        """Extract LHS variable from assignment"""
        node = parse_result.nodes[node_idx]
        if node.children_indices:
            first_child = parse_result.nodes[node.children_indices[0]]
            if first_child.node_type == 'identifier':
                return first_child.text.strip()
            elif first_child.node_type in ('pointer_expression', 'subscript_expression', 
                                            'field_expression'):
                return self._find_identifier_in_subtree(parse_result, node.children_indices[0])
        return None

    def _extract_rhs_vars(self, parse_result: ParseResult, node_idx: int) -> List[str]:
        """Extract RHS variables from assignment"""
        vars_found: List[str] = []
        node = parse_result.nodes[node_idx]
        
        if len(node.children_indices) >= 2:
            rhs_idx = node.children_indices[-1]
            self._collect_identifiers(parse_result, rhs_idx, vars_found)
        
        return vars_found

    def _extract_update_var(self, parse_result: ParseResult, node_idx: int) -> Optional[str]:
        """Extract variable from update expression (++/--) """
        node = parse_result.nodes[node_idx]
        return self._find_identifier_in_subtree(parse_result, node_idx)

    def _extract_return_vars(self, parse_result: ParseResult, node_idx: int) -> List[str]:
        """Extract variables from return statement"""
        vars_found: List[str] = []
        node = parse_result.nodes[node_idx]
        
        for child_idx in node.children_indices:
            self._collect_identifiers(parse_result, child_idx, vars_found)
        
        return vars_found

    def _extract_pointer_access(self, parse_result: ParseResult, 
                                node_idx: int) -> Optional[Tuple[str, UseType]]:
        """Extract pointer access info (* or &)"""
        node = parse_result.nodes[node_idx]
        text = node.text.strip()
        
        var_name = self._find_identifier_in_subtree(parse_result, node_idx)
        if not var_name:
            return None
        
        if text.startswith('&'):
            return (var_name, UseType.ADDRESS)
        elif text.startswith('*'):
            return (var_name, UseType.DEREF)
        
        return None

    def _extract_func_name(self, parse_result: ParseResult, node_idx: int) -> str:
        """Extract function name from call expression"""
        node = parse_result.nodes[node_idx]
        for child_idx in node.children_indices:
            child = parse_result.nodes[child_idx]
            if child.node_type == 'identifier':
                return child.text.strip()
            elif child.node_type == 'field_expression':
                return self._find_identifier_in_subtree(parse_result, child_idx) or ''
        return ''

    def _extract_call_args(self, parse_result: ParseResult, node_idx: int) -> List[str]:
        """Extract argument variable names from call"""
        args: List[str] = []
        node = parse_result.nodes[node_idx]
        
        for child_idx in node.children_indices:
            child = parse_result.nodes[child_idx]
            if child.node_type == 'argument_list':
                for arg_idx in child.children_indices:
                    arg_node = parse_result.nodes[arg_idx]
                    if arg_node.node_type == 'identifier':
                        args.append(arg_node.text.strip())
                    elif arg_node.node_type not in ('(', ')', ','):
                        self._collect_identifiers(parse_result, arg_idx, args)
        
        return args

    def _find_identifier_in_subtree(self, parse_result: ParseResult, node_idx: int) -> Optional[str]:
        """Find first identifier in subtree"""
        node = parse_result.nodes[node_idx]
        if node.node_type == 'identifier':
            return node.text.strip()
        
        for child_idx in node.children_indices:
            result = self._find_identifier_in_subtree(parse_result, child_idx)
            if result:
                return result
        
        return None

    def _collect_identifiers(self, parse_result: ParseResult, node_idx: int, 
                            result: List[str]) -> None:
        """Collect all identifiers in subtree"""
        node = parse_result.nodes[node_idx]
        
        if node.node_type == 'identifier':
            text = node.text.strip()
            if text and not text.isupper() and text not in result:
                result.append(text)
            return
        
        for child_idx in node.children_indices:
            self._collect_identifiers(parse_result, child_idx, result)


def serialize_dfg(dfg: DFG) -> dict:
    """Serialize DFG for saving"""
    return {
        'nodes': [
            {
                'var_name': n.var_name,
                'line': n.line,
                'col': n.col,
                'access_type': n.access_type.value,
                'node_index': n.node_index,
                'context': n.context,
            }
            for n in dfg.nodes
        ],
        'edges': [
            {
                'from_idx': e.from_idx,
                'to_idx': e.to_idx,
                'edge_type': e.edge_type,
            }
            for e in dfg.edges
        ],
        'var_defs': dfg.var_defs,
        'var_uses': dfg.var_uses,
    }


def deserialize_dfg(data: dict) -> DFG:
    """Deserialize DFG from dict"""
    nodes = [
        VariableAccess(
            var_name=n['var_name'],
            line=n['line'],
            col=n['col'],
            access_type=UseType(n['access_type']),
            node_index=n['node_index'],
            context=n.get('context', ''),
        )
        for n in data['nodes']
    ]
    
    edges = [
        DFGEdge(
            from_idx=e['from_idx'],
            to_idx=e['to_idx'],
            edge_type=e['edge_type'],
        )
        for e in data['edges']
    ]
    
    return DFG(
        nodes=nodes,
        edges=edges,
        var_defs=data['var_defs'],
        var_uses=data['var_uses'],
    )


def build_dfg(parse_result: ParseResult, 
              focus_lines: Optional[List[int]] = None,
              scope_limit: int = 50) -> Optional[DFG]:
    """Convenience function to build DFG from ParseResult"""
    builder = DFGBuilder(scope_limit=scope_limit)
    return builder.build(parse_result, focus_lines)
