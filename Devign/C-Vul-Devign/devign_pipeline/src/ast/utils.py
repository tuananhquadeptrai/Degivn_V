"""AST utility functions"""

from typing import List, Dict, Any, Set, Optional
from .parser import ASTParser, CFamilyParser, ParseResult, ASTNode, Language


def get_node_by_line(result: ParseResult, line: int) -> List[ASTNode]:
    """Get all nodes that span the given line"""
    if not result.nodes or result.root_index < 0:
        return []
    
    matching = []
    for node in result.nodes:
        if node.start_line <= line <= node.end_line:
            matching.append(node)
    return matching


def get_nodes_in_range(result: ParseResult, start_line: int, end_line: int) -> List[ASTNode]:
    """Get nodes within line range"""
    if not result.nodes or result.root_index < 0:
        return []
    
    matching = []
    for node in result.nodes:
        if node.end_line >= start_line and node.start_line <= end_line:
            matching.append(node)
    return matching


def find_nodes_by_type(result: ParseResult, node_types: List[str]) -> List[ASTNode]:
    """Find all nodes of specific types (e.g., 'function_definition', 'if_statement')"""
    if not result.nodes or result.root_index < 0:
        return []
    
    type_set = set(node_types)
    return [node for node in result.nodes if node.node_type in type_set]


def get_function_nodes(result: ParseResult) -> List[ASTNode]:
    """Extract function definitions"""
    function_types = [
        'function_definition',
        'function_declarator',
        'method_definition',      
        'lambda_expression',      
    ]
    return find_nodes_by_type(result, function_types)


def get_control_flow_nodes(result: ParseResult) -> List[ASTNode]:
    """Get if/while/for/switch nodes"""
    control_types = [
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
        'try_statement',          
        'catch_clause',           
        'throw_statement',        
        'for_range_loop',         
    ]
    return find_nodes_by_type(result, control_types)


def map_vul_lines_to_nodes(result: ParseResult, vul_lines: List[int]) -> List[int]:
    """Map vulnerability line numbers to AST node indices"""
    if not result.nodes or result.root_index < 0 or not vul_lines:
        return []
    
    vul_set = set(vul_lines)
    indices = []
    
    for idx, node in enumerate(result.nodes):
        for line in range(node.start_line, node.end_line + 1):
            if line in vul_set:
                indices.append(idx)
                break
    
    return indices


def get_smallest_nodes_for_lines(result: ParseResult, lines: List[int]) -> Dict[int, Optional[int]]:
    """
    Get the smallest (most specific) node index for each line.
    Returns dict mapping line -> node_index (or None if no node found).
    """
    if not result.nodes or result.root_index < 0:
        return {line: None for line in lines}
    
    line_to_node: Dict[int, Optional[int]] = {}
    
    for line in lines:
        best_idx: Optional[int] = None
        best_size = float('inf')
        
        for idx, node in enumerate(result.nodes):
            if node.start_line <= line <= node.end_line:
                size = node.end_byte - node.start_byte
                if size < best_size:
                    best_size = size
                    best_idx = idx
        
        line_to_node[line] = best_idx
    
    return line_to_node


def get_parent_chain(result: ParseResult, node_index: int) -> List[int]:
    """Get list of parent indices from node up to root"""
    if not result.nodes or node_index < 0 or node_index >= len(result.nodes):
        return []
    
    chain = []
    current = node_index
    visited = set()
    
    while current is not None and current not in visited:
        visited.add(current)
        parent_idx = result.nodes[current].parent_index
        if parent_idx is not None:
            chain.append(parent_idx)
        current = parent_idx
    
    return chain


def get_containing_function(result: ParseResult, node_index: int) -> Optional[int]:
    """Get the function definition node that contains the given node"""
    function_types = {'function_definition', 'method_definition'}
    
    chain = get_parent_chain(result, node_index)
    for idx in chain:
        if result.nodes[idx].node_type in function_types:
            return idx
    
    node = result.nodes[node_index] if 0 <= node_index < len(result.nodes) else None
    if node and node.node_type in function_types:
        return node_index
    
    return None


def serialize_ast(result: ParseResult) -> dict:
    """Serialize ParseResult to dict for saving"""
    return {
        'nodes': [
            {
                'node_type': node.node_type,
                'text': node.text,
                'start_line': node.start_line,
                'start_col': node.start_col,
                'end_line': node.end_line,
                'end_col': node.end_col,
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'children_indices': node.children_indices,
                'parent_index': node.parent_index,
                'is_error': node.is_error,
            }
            for node in result.nodes
        ],
        'root_index': result.root_index,
        'has_errors': result.has_errors,
        'error_count': result.error_count,
        'source_code': result.source_code,
    }


def deserialize_ast(data: dict) -> ParseResult:
    """Deserialize dict to ParseResult"""
    nodes = [
        ASTNode(
            node_type=n['node_type'],
            text=n['text'],
            start_line=n['start_line'],
            start_col=n['start_col'],
            end_line=n['end_line'],
            end_col=n['end_col'],
            start_byte=n['start_byte'],
            end_byte=n['end_byte'],
            children_indices=n.get('children_indices', []),
            parent_index=n.get('parent_index'),
            is_error=n.get('is_error', False),
        )
        for n in data.get('nodes', [])
    ]
    
    return ParseResult(
        nodes=nodes,
        root_index=data.get('root_index', -1),
        has_errors=data.get('has_errors', False),
        error_count=data.get('error_count', 0),
        source_code=data.get('source_code', ''),
    )


def get_function_calls(code: str) -> List[str]:
    """Extract all function call names from code"""
    parser = ASTParser()
    root = parser.get_root(code)
    if not root:
        return []
    
    calls = []
    for node in parser.walk(root):
        if node.type == 'call_expression':
            func_node = node.child_by_field_name('function')
            if func_node:
                calls.append(func_node.text.decode('utf-8'))
    return calls


def get_identifiers(code: str) -> Set[str]:
    """Extract all identifiers from code"""
    parser = ASTParser()
    root = parser.get_root(code)
    if not root:
        return set()
    
    identifiers = set()
    for node in parser.walk(root):
        if node.type == 'identifier':
            identifiers.add(node.text.decode('utf-8'))
    return identifiers


def ast_to_dict(node: Any, code_bytes: bytes = None) -> Dict[str, Any]:
    """Convert AST node to dictionary representation"""
    result = {
        'type': node.type,
        'start_point': node.start_point,
        'end_point': node.end_point,
    }
    
    if code_bytes and node.child_count == 0:
        result['text'] = node.text.decode('utf-8')
    
    if node.children:
        result['children'] = [
            ast_to_dict(child, code_bytes) for child in node.children
        ]
    
    return result


def get_node_text_from_source(result: ParseResult, node_index: int) -> str:
    """Extract node text from source code using byte offsets"""
    if not result.nodes or node_index < 0 or node_index >= len(result.nodes):
        return ''
    
    node = result.nodes[node_index]
    try:
        source_bytes = result.source_code.encode('utf-8')
        return source_bytes[node.start_byte:node.end_byte].decode('utf-8')
    except (UnicodeDecodeError, IndexError):
        return node.text


def count_node_types(result: ParseResult) -> Dict[str, int]:
    """Count occurrences of each node type in the AST"""
    counts: Dict[str, int] = {}
    for node in result.nodes:
        counts[node.node_type] = counts.get(node.node_type, 0) + 1
    return counts


def get_leaf_nodes(result: ParseResult) -> List[ASTNode]:
    """Get all leaf nodes (nodes with no children)"""
    if not result.nodes:
        return []
    return [node for node in result.nodes if not node.children_indices]


def get_depth(result: ParseResult, node_index: int) -> int:
    """Get depth of a node in the tree (root = 0)"""
    return len(get_parent_chain(result, node_index))


def get_max_depth(result: ParseResult) -> int:
    """Get maximum depth of the AST"""
    if not result.nodes or result.root_index < 0:
        return 0
    return max(get_depth(result, i) for i in range(len(result.nodes)))
