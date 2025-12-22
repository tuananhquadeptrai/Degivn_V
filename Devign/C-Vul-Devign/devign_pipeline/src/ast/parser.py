"""AST parsing for C/C++ code using tree-sitter"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
from enum import Enum
import re


class Language(Enum):
    C = 'c'
    CPP = 'cpp'


@dataclass
class ASTNode:
    node_type: str
    text: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_byte: int
    end_byte: int
    children_indices: List[int] = field(default_factory=list)
    parent_index: Optional[int] = None
    is_error: bool = False


@dataclass
class ParseResult:
    nodes: List[ASTNode]
    root_index: int
    has_errors: bool
    error_count: int
    source_code: str


class CFamilyParser:
    """Parse C/C++ code into AST using tree-sitter"""
    
    _CPP_PATTERNS = [
        r'\bclass\s+\w+',
        r'\btemplate\s*<',
        r'\bnamespace\s+\w+',
        r'\bpublic\s*:',
        r'\bprivate\s*:',
        r'\bprotected\s*:',
        r'\bvirtual\s+',
        r'\b(std|boost)::',
        r'\bnew\s+\w+',
        r'\bdelete\s+',
        r'\bcout\b',
        r'\bcin\b',
        r'\bconst_cast\b',
        r'\bstatic_cast\b',
        r'\bdynamic_cast\b',
        r'\breinterpret_cast\b',
        r'\bnullptr\b',
        r'\bauto\s+\w+\s*=',
        r'::\w+',
        r'\busing\s+namespace\b',
        r'\boverride\b',
        r'\bfinal\b',
    ]
    
    _CPP_PROJECTS = [
        'qemu', 'chrome', 'chromium', 'firefox', 'webkit',
        'llvm', 'clang', 'gcc', 'boost'
    ]
    
    def __init__(self, language: Language = Language.C):
        """Initialize tree-sitter parser for C or C++"""
        self.language = language
        self._parser = None
        self._ts_language = None
        self._initialized = False
        
    def _ensure_initialized(self) -> None:
        """Lazy initialization of tree-sitter"""
        if self._initialized:
            return
            
        try:
            from tree_sitter import Language as TSLanguage, Parser
            
            if self.language == Language.CPP:
                import tree_sitter_cpp as ts_cpp
                self._ts_language = TSLanguage(ts_cpp.language())
            else:
                import tree_sitter_c as ts_c
                self._ts_language = TSLanguage(ts_c.language())
            
            self._parser = Parser(self._ts_language)
            self._initialized = True
            
        except ImportError as e:
            missing = []
            if 'tree_sitter' in str(e) or 'tree-sitter' in str(e):
                missing.append('tree-sitter')
            if 'tree_sitter_c' in str(e):
                missing.append('tree-sitter-c')
            if 'tree_sitter_cpp' in str(e):
                missing.append('tree-sitter-cpp')
            
            if not missing:
                missing = ['tree-sitter', 'tree-sitter-c', 'tree-sitter-cpp']
                
            raise ImportError(
                f"Required packages missing. Install with: pip install {' '.join(missing)}"
            )
    
    def _convert_tree_to_nodes(self, tree: Any, source_code: str) -> ParseResult:
        """Convert tree-sitter tree to flat list of ASTNodes"""
        nodes: List[ASTNode] = []
        error_count = 0
        
        def traverse(ts_node: Any, parent_idx: Optional[int] = None) -> int:
            nonlocal error_count
            
            current_idx = len(nodes)
            
            is_error = ts_node.type == 'ERROR' or ts_node.is_missing
            if is_error:
                error_count += 1
            
            try:
                text = ts_node.text.decode('utf-8') if ts_node.text else ''
            except (UnicodeDecodeError, AttributeError):
                text = ''
            
            node = ASTNode(
                node_type=ts_node.type,
                text=text,
                start_line=ts_node.start_point[0] + 1,
                start_col=ts_node.start_point[1],
                end_line=ts_node.end_point[0] + 1,
                end_col=ts_node.end_point[1],
                start_byte=ts_node.start_byte,
                end_byte=ts_node.end_byte,
                children_indices=[],
                parent_index=parent_idx,
                is_error=is_error,
            )
            nodes.append(node)
            
            for child in ts_node.children:
                child_idx = traverse(child, current_idx)
                nodes[current_idx].children_indices.append(child_idx)
            
            return current_idx
        
        root_idx = traverse(tree.root_node)
        
        return ParseResult(
            nodes=nodes,
            root_index=root_idx,
            has_errors=error_count > 0,
            error_count=error_count,
            source_code=source_code,
        )
    
    def parse(self, code: str) -> Optional[ParseResult]:
        """
        Parse code and return ParseResult.
        Returns None if parse completely fails.
        """
        if not code or not code.strip():
            return ParseResult(
                nodes=[],
                root_index=-1,
                has_errors=False,
                error_count=0,
                source_code=code,
            )
        
        self._ensure_initialized()
        
        try:
            code_bytes = code.encode('utf-8')
            tree = self._parser.parse(code_bytes)
            
            if tree is None:
                return None
                
            return self._convert_tree_to_nodes(tree, code)
            
        except Exception:
            return None
    
    def parse_with_fallback(self, code: str) -> ParseResult:
        """
        Try C++ first, fallback to C if errors.
        Returns the result with fewer errors.
        """
        if not code or not code.strip():
            return ParseResult(
                nodes=[],
                root_index=-1,
                has_errors=False,
                error_count=0,
                source_code=code,
            )
        
        original_lang = self.language
        original_initialized = self._initialized
        original_parser = self._parser
        original_ts_language = self._ts_language
        
        try:
            self.language = Language.CPP
            self._initialized = False
            cpp_result = self.parse(code)
            
            if cpp_result and not cpp_result.has_errors:
                return cpp_result
            
            self.language = Language.C
            self._initialized = False
            c_result = self.parse(code)
            
            if c_result is None and cpp_result is None:
                return ParseResult(
                    nodes=[],
                    root_index=-1,
                    has_errors=True,
                    error_count=1,
                    source_code=code,
                )
            
            if c_result is None:
                return cpp_result
            if cpp_result is None:
                return c_result
                
            if c_result.error_count <= cpp_result.error_count:
                return c_result
            return cpp_result
            
        finally:
            self.language = original_lang
            self._initialized = original_initialized
            self._parser = original_parser
            self._ts_language = original_ts_language
    
    @staticmethod
    def detect_language(code: str, project: str = '') -> Language:
        """
        Detect C vs C++ from code patterns or project name.
        """
        project_lower = project.lower()
        for cpp_proj in CFamilyParser._CPP_PROJECTS:
            if cpp_proj in project_lower:
                return Language.CPP
        
        for pattern in CFamilyParser._CPP_PATTERNS:
            if re.search(pattern, code):
                return Language.CPP
        
        if re.search(r'#include\s*<\w+>', code):
            if not re.search(r'#include\s*<(stdio|stdlib|string|math|ctype|time|assert)\.h>', code):
                return Language.CPP
        
        return Language.C
    
    def get_root(self, code: str) -> Optional[Any]:
        """Get root node of AST (returns tree-sitter node for compatibility)"""
        self._ensure_initialized()
        try:
            tree = self._parser.parse(code.encode('utf-8'))
            return tree.root_node if tree else None
        except Exception:
            return None
    
    def walk(self, node: Any) -> List[Any]:
        """Walk AST and return all nodes (for compatibility)"""
        nodes = [node]
        for child in node.children:
            nodes.extend(self.walk(child))
        return nodes


class ASTParser(CFamilyParser):
    """Backward-compatible alias for CFamilyParser"""
    
    def __init__(self):
        super().__init__(Language.C)
        self.parser = None  
        
    def _ensure_initialized(self) -> None:
        super()._ensure_initialized()
        self.parser = self._parser


def parse_code(code: str) -> Optional[Any]:
    """Convenience function to parse code"""
    parser = ASTParser()
    return parser.get_root(code)
