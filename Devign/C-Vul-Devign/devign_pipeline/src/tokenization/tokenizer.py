"""C/C++ code tokenization using tree-sitter with regex fallback"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from collections import Counter
import re

import sys
sys.path.insert(0, '/media/hdi/Hdii/Work/C Vul Devign/devign_pipeline')
from src.ast.parser import CFamilyParser, ParseResult, Language


class TokenType(Enum):
    KEYWORD = 'keyword'
    IDENTIFIER = 'identifier'
    TYPE = 'type'
    OPERATOR = 'operator'
    LITERAL_NUM = 'literal_num'
    LITERAL_STR = 'literal_str'
    LITERAL_CHAR = 'literal_char'
    PUNCTUATION = 'punct'
    PREPROCESSOR = 'preproc'
    COMMENT = 'comment'
    UNKNOWN = 'unknown'


C_KEYWORDS = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
    'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
    'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while',
    'inline', 'restrict', '_Bool', '_Complex', '_Imaginary',
    'class', 'public', 'private', 'protected', 'virtual', 'override',
    'template', 'typename', 'namespace', 'using', 'new', 'delete',
    'try', 'catch', 'throw', 'nullptr', 'bool', 'true', 'false',
    'const_cast', 'static_cast', 'dynamic_cast', 'reinterpret_cast',
    'final', 'noexcept', 'explicit', 'friend', 'mutable', 'operator',
    'this', 'constexpr', 'decltype', 'alignas', 'alignof', 'static_assert',
}

C_TYPES = {
    'int', 'char', 'float', 'double', 'void', 'long', 'short',
    'unsigned', 'signed', 'bool', 'size_t', 'ssize_t', 'ptrdiff_t',
    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
    'int8_t', 'int16_t', 'int32_t', 'int64_t',
    'intptr_t', 'uintptr_t', 'wchar_t', 'char16_t', 'char32_t',
    'FILE', 'NULL', 'nullptr_t',
}

C_OPERATORS = {
    '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=',
    '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--',
    '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=',
    '->', '.', '::', '?', ':', '...', 'sizeof', 'alignof',
}

C_PUNCTUATION = {'{', '}', '(', ')', '[', ']', ';', ',', '#'}

TS_TYPE_MAP = {
    'identifier': TokenType.IDENTIFIER,
    'type_identifier': TokenType.TYPE,
    'primitive_type': TokenType.TYPE,
    'sized_type_specifier': TokenType.TYPE,
    'number_literal': TokenType.LITERAL_NUM,
    'string_literal': TokenType.LITERAL_STR,
    'string_content': TokenType.LITERAL_STR,
    'char_literal': TokenType.LITERAL_CHAR,
    'character': TokenType.LITERAL_CHAR,
    'comment': TokenType.COMMENT,
    'preproc_directive': TokenType.PREPROCESSOR,
    'preproc_include': TokenType.PREPROCESSOR,
    'preproc_def': TokenType.PREPROCESSOR,
    'preproc_ifdef': TokenType.PREPROCESSOR,
    'preproc_ifndef': TokenType.PREPROCESSOR,
    'preproc_else': TokenType.PREPROCESSOR,
    'preproc_endif': TokenType.PREPROCESSOR,
    'preproc_if': TokenType.PREPROCESSOR,
    'preproc_arg': TokenType.IDENTIFIER,
    'system_lib_string': TokenType.LITERAL_STR,
}


@dataclass
class Token:
    text: str
    token_type: TokenType
    line: int
    col: int
    start_byte: int = 0
    end_byte: int = 0
    
    def __repr__(self) -> str:
        return f"Token({self.text!r}, {self.token_type.value}, L{self.line})"


class CTokenizer:
    """Tokenizer for C/C++ code using tree-sitter with regex fallback"""
    
    _REGEX_PATTERNS = [
        (r'//[^\n]*', TokenType.COMMENT),
        (r'/\*[\s\S]*?\*/', TokenType.COMMENT),
        (r'#\s*\w+[^\n]*', TokenType.PREPROCESSOR),
        (r'"(?:[^"\\]|\\.)*"', TokenType.LITERAL_STR),
        (r"'(?:[^'\\]|\\.)*'", TokenType.LITERAL_CHAR),
        (r'0[xX][0-9a-fA-F]+[uUlL]*', TokenType.LITERAL_NUM),
        (r'0[bB][01]+[uUlL]*', TokenType.LITERAL_NUM),
        (r'\d+\.?\d*(?:[eE][+-]?\d+)?[fFlLuU]*', TokenType.LITERAL_NUM),
        (r'\.\d+(?:[eE][+-]?\d+)?[fFlL]*', TokenType.LITERAL_NUM),
        (r'\.\.\.', TokenType.OPERATOR),
        (r'::', TokenType.OPERATOR),
        (r'->', TokenType.OPERATOR),
        (r'\+\+|--', TokenType.OPERATOR),
        (r'<<=|>>=', TokenType.OPERATOR),
        (r'<<|>>', TokenType.OPERATOR),
        (r'<=|>=|==|!=', TokenType.OPERATOR),
        (r'&&|\|\|', TokenType.OPERATOR),
        (r'[+\-*/%&|^~!=<>]=', TokenType.OPERATOR),
        (r'[+\-*/%&|^~!=<>?:]', TokenType.OPERATOR),
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),
        (r'[{}()\[\];,.]', TokenType.PUNCTUATION),
    ]
    
    def __init__(self, include_comments: bool = False):
        self.include_comments = include_comments
        self._parser = None
        self._parser_initialized = False
        self._compiled_patterns = [
            (re.compile(p, re.MULTILINE), t) for p, t in self._REGEX_PATTERNS
        ]
        
    def _get_parser(self) -> CFamilyParser:
        if not self._parser_initialized:
            self._parser = CFamilyParser()
            self._parser_initialized = True
        return self._parser
        
    def tokenize(self, code: str) -> List[Token]:
        """Tokenize C/C++ code using tree-sitter. Falls back to regex if parsing fails."""
        if not code or not code.strip():
            return []
            
        try:
            parser = self._get_parser()
            parse_result = parser.parse_with_fallback(code)
            
            if parse_result and parse_result.nodes:
                tokens = self._extract_tokens_from_tree(parse_result)
                if tokens:
                    if not self.include_comments:
                        tokens = [t for t in tokens if t.token_type != TokenType.COMMENT]
                    return tokens
        except Exception:
            pass
            
        return self.tokenize_regex(code)
        
    def tokenize_with_types(self, code: str) -> List[Tuple[str, TokenType]]:
        """Return simplified (text, type) pairs"""
        tokens = self.tokenize(code)
        return [(t.text, t.token_type) for t in tokens]
        
    def tokenize_regex(self, code: str) -> List[Token]:
        """Fallback regex-based tokenizer. Less accurate but works without parser."""
        if not code:
            return []
            
        tokens = []
        lines = code.split('\n')
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line) + 1)
        
        pos = 0
        while pos < len(code):
            if code[pos].isspace():
                pos += 1
                continue
                
            matched = False
            for pattern, token_type in self._compiled_patterns:
                match = pattern.match(code, pos)
                if match:
                    text = match.group()
                    start_byte = pos
                    end_byte = match.end()
                    
                    line = 1
                    for i, offset in enumerate(line_offsets):
                        if offset > start_byte:
                            line = i
                            break
                    else:
                        line = len(line_offsets)
                    col = start_byte - line_offsets[line - 1]
                    
                    actual_type = self._classify_token_regex(text, token_type)
                    
                    if actual_type == TokenType.COMMENT and not self.include_comments:
                        pos = end_byte
                        matched = True
                        break
                    
                    tokens.append(Token(
                        text=text,
                        token_type=actual_type,
                        line=line,
                        col=col,
                        start_byte=start_byte,
                        end_byte=end_byte,
                    ))
                    pos = end_byte
                    matched = True
                    break
                    
            if not matched:
                pos += 1
                
        return tokens
        
    def _classify_token_regex(self, text: str, default_type: TokenType) -> TokenType:
        """Classify token for regex tokenizer"""
        if default_type != TokenType.IDENTIFIER:
            return default_type
            
        if text in C_KEYWORDS:
            if text in C_TYPES:
                return TokenType.TYPE
            return TokenType.KEYWORD
        if text in C_TYPES:
            return TokenType.TYPE
        if text in C_OPERATORS:
            return TokenType.OPERATOR
            
        return TokenType.IDENTIFIER
        
    def _classify_token(self, text: str, ts_type: str) -> TokenType:
        """Classify token based on tree-sitter node type and text"""
        if ts_type in TS_TYPE_MAP:
            mapped = TS_TYPE_MAP[ts_type]
            if mapped == TokenType.IDENTIFIER and text in C_KEYWORDS:
                if text in C_TYPES:
                    return TokenType.TYPE
                return TokenType.KEYWORD
            return mapped
            
        if 'comment' in ts_type:
            return TokenType.COMMENT
        if 'preproc' in ts_type:
            return TokenType.PREPROCESSOR
        if 'string' in ts_type:
            return TokenType.LITERAL_STR
        if 'char' in ts_type and 'literal' in ts_type:
            return TokenType.LITERAL_CHAR
        if 'number' in ts_type or 'literal' in ts_type:
            return TokenType.LITERAL_NUM
        if 'type' in ts_type:
            return TokenType.TYPE
            
        if text in C_KEYWORDS:
            if text in C_TYPES:
                return TokenType.TYPE
            return TokenType.KEYWORD
        if text in C_TYPES:
            return TokenType.TYPE
        if text in C_OPERATORS:
            return TokenType.OPERATOR
        if text in C_PUNCTUATION:
            return TokenType.PUNCTUATION
            
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', text):
            return TokenType.IDENTIFIER
        if re.match(r'^[\d.]', text):
            return TokenType.LITERAL_NUM
            
        if len(text) <= 3 and not text.isalnum():
            return TokenType.OPERATOR
            
        if text.startswith('#'):
            return TokenType.PREPROCESSOR
            
        return TokenType.UNKNOWN
        
    def _extract_tokens_from_tree(self, parse_result: ParseResult) -> List[Token]:
        """Walk tree-sitter tree and extract leaf tokens"""
        tokens = []
        source = parse_result.source_code
        
        for node in parse_result.nodes:
            if node.children_indices:
                continue
                
            text = node.text
            if not text or not text.strip():
                continue
                
            if node.node_type in ('ERROR', 'MISSING'):
                continue
                
            token_type = self._classify_token(text, node.node_type)
            
            tokens.append(Token(
                text=text,
                token_type=token_type,
                line=node.start_line,
                col=node.start_col,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
            ))
            
        tokens.sort(key=lambda t: (t.start_byte, t.line, t.col))
        
        seen = set()
        unique_tokens = []
        for t in tokens:
            key = (t.start_byte, t.end_byte)
            if key not in seen:
                seen.add(key)
                unique_tokens.append(t)
                
        return unique_tokens
        
    def filter_tokens(self, tokens: List[Token],
                      exclude_types: Optional[List[TokenType]] = None) -> List[Token]:
        """Filter out unwanted token types (e.g., comments)"""
        if exclude_types is None:
            exclude_types = [TokenType.COMMENT]
        exclude_set = set(exclude_types)
        return [t for t in tokens if t.token_type not in exclude_set]
        
    def get_token_stats(self, tokens: List[Token]) -> Dict[str, Any]:
        """Get statistics about tokens"""
        type_counts = Counter(t.token_type.value for t in tokens)
        unique_tokens = set(t.text for t in tokens)
        
        return {
            'total_tokens': len(tokens),
            'unique_tokens': len(unique_tokens),
            'type_distribution': dict(type_counts),
            'avg_token_length': sum(len(t.text) for t in tokens) / len(tokens) if tokens else 0,
            'keywords': sum(1 for t in tokens if t.token_type == TokenType.KEYWORD),
            'identifiers': sum(1 for t in tokens if t.token_type == TokenType.IDENTIFIER),
            'literals': sum(1 for t in tokens if t.token_type in {
                TokenType.LITERAL_NUM, TokenType.LITERAL_STR, TokenType.LITERAL_CHAR
            }),
        }


def _tokenize_single(args: Tuple[str, bool]) -> List[Token]:
    """Helper for multiprocessing"""
    code, include_comments = args
    tokenizer = CTokenizer(include_comments=include_comments)
    return tokenizer.tokenize(code)


def tokenize_batch(codes: List[str], include_comments: bool = False,
                   n_jobs: int = 4) -> List[List[Token]]:
    """Batch tokenization with multiprocessing"""
    if not codes:
        return []
        
    if len(codes) == 1 or n_jobs <= 1:
        tokenizer = CTokenizer(include_comments=include_comments)
        return [tokenizer.tokenize(code) for code in codes]
        
    from multiprocessing import Pool
    
    args = [(code, include_comments) for code in codes]
    
    with Pool(processes=min(n_jobs, len(codes))) as pool:
        results = pool.map(_tokenize_single, args)
        
    return results


class Tokenizer(CTokenizer):
    """Backward-compatible alias"""
    
    def __init__(self, include_whitespace: bool = False):
        super().__init__(include_comments=False)
        self.include_whitespace = include_whitespace
        
    def tokenize(self, code: str) -> List[str]:
        """Return just token texts for backward compatibility"""
        tokens = super().tokenize(code)
        return [t.text for t in tokens]
        
    def tokenize_with_types(self, code: str) -> List[Tuple[str, str]]:
        """Return (text, type_string) pairs for backward compatibility"""
        tokens = CTokenizer.tokenize(self, code)
        type_map = {
            TokenType.KEYWORD: 'KEYWORD',
            TokenType.IDENTIFIER: 'IDENTIFIER',
            TokenType.TYPE: 'KEYWORD',
            TokenType.OPERATOR: 'OPERATOR',
            TokenType.LITERAL_NUM: 'NUMBER',
            TokenType.LITERAL_STR: 'STRING',
            TokenType.LITERAL_CHAR: 'CHAR',
            TokenType.PUNCTUATION: 'PUNCTUATION',
            TokenType.PREPROCESSOR: 'KEYWORD',
            TokenType.COMMENT: 'COMMENT',
            TokenType.UNKNOWN: 'IDENTIFIER',
        }
        return [(t.text, type_map.get(t.token_type, 'IDENTIFIER')) for t in tokens]


def tokenize_code(code: str) -> List[str]:
    """Convenience function to tokenize code"""
    return Tokenizer().tokenize(code)
