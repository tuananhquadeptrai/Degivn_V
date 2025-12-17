"""C code normalization for vulnerability detection"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum, auto
import re
from multiprocessing import Pool
from functools import partial

from .tokenizer import Tokenizer


C_KEYWORDS = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
    'inline', 'int', 'long', 'register', 'restrict', 'return', 'short',
    'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
    'unsigned', 'void', 'volatile', 'while', '_Bool', '_Complex', '_Imaginary',
    'NULL', 'true', 'false', 'nullptr'
}

C_TYPES = {
    'int', 'char', 'float', 'double', 'void', 'long', 'short', 'signed',
    'unsigned', 'bool', '_Bool', 'size_t', 'ssize_t', 'int8_t', 'int16_t',
    'int32_t', 'int64_t', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
    'intptr_t', 'uintptr_t', 'ptrdiff_t', 'wchar_t', 'FILE', 'time_t',
    'off_t', 'pid_t', 'uid_t', 'gid_t'
}

COMMON_STDLIB_FUNCS = {
    'printf', 'scanf', 'malloc', 'calloc', 'realloc', 'free',
    'memcpy', 'memset', 'memmove', 'memcmp', 'strlen', 'strcpy',
    'strncpy', 'strcat', 'strncat', 'strcmp', 'strncmp', 'strchr',
    'strrchr', 'strstr', 'sprintf', 'snprintf', 'sscanf', 'fprintf',
    'fscanf', 'fopen', 'fclose', 'fread', 'fwrite', 'fgets', 'fputs',
    'fseek', 'ftell', 'rewind', 'feof', 'ferror', 'fflush',
    'exit', 'abort', 'atexit', 'atoi', 'atol', 'atof', 'strtol',
    'strtoul', 'strtod', 'rand', 'srand', 'abs', 'labs', 'qsort',
    'bsearch', 'getchar', 'putchar', 'gets', 'puts', 'getenv',
    'system', 'assert', 'perror', 'errno'
}


class TokenType(Enum):
    KEYWORD = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    HEX = auto()
    STRING = auto()
    CHAR = auto()
    OPERATOR = auto()
    PUNCTUATION = auto()
    COMMENT = auto()
    WHITESPACE = auto()
    TYPE = auto()
    FUNCTION = auto()
    VARIABLE = auto()
    UNKNOWN = auto()


@dataclass
class Token:
    value: str
    type: TokenType
    position: int = 0
    
    def __repr__(self):
        return f"Token({self.value!r}, {self.type.name})"


@dataclass
class NormalizationMaps:
    """Stores mappings from original to normalized names"""
    var_map: Dict[str, str] = field(default_factory=dict)
    func_map: Dict[str, str] = field(default_factory=dict)
    type_map: Dict[str, str] = field(default_factory=dict)
    var_counter: int = 0
    func_counter: int = 0
    type_counter: int = 0
    
    def get_or_add_var(self, name: str) -> str:
        """Get normalized var name or add new mapping"""
        if name not in self.var_map:
            self.var_map[name] = f'VAR_{self.var_counter}'
            self.var_counter += 1
        return self.var_map[name]
    
    def get_or_add_func(self, name: str) -> str:
        """Get normalized func name or add new mapping"""
        if name not in self.func_map:
            self.func_map[name] = f'FUNC_{self.func_counter}'
            self.func_counter += 1
        return self.func_map[name]
    
    def get_or_add_type(self, name: str) -> str:
        """Get normalized type name or add new mapping"""
        if name not in self.type_map:
            self.type_map[name] = f'TYPE_{self.type_counter}'
            self.type_counter += 1
        return self.type_map[name]
    
    def reverse_map(self) -> Dict[str, str]:
        """Get reverse mapping (normalized -> original)"""
        reverse = {}
        for orig, norm in self.var_map.items():
            reverse[norm] = orig
        for orig, norm in self.func_map.items():
            reverse[norm] = orig
        for orig, norm in self.type_map.items():
            reverse[norm] = orig
        return reverse
    
    def copy(self) -> 'NormalizationMaps':
        """Create a deep copy of the maps"""
        new_maps = NormalizationMaps()
        new_maps.var_map = self.var_map.copy()
        new_maps.func_map = self.func_map.copy()
        new_maps.type_map = self.type_map.copy()
        new_maps.var_counter = self.var_counter
        new_maps.func_counter = self.func_counter
        new_maps.type_counter = self.type_counter
        return new_maps


NORM_NUM = 'NUM'
NORM_STR = 'STR'
NORM_CHAR = 'CHAR'
NORM_FLOAT = 'FLOAT'


class CodeNormalizer:
    def __init__(self,
                 normalize_vars: bool = True,
                 normalize_funcs: bool = True,
                 normalize_literals: bool = True,
                 normalize_types: bool = False,
                 preserve_keywords: bool = True,
                 preserve_stdlib: bool = True):
        self.normalize_vars = normalize_vars
        self.normalize_funcs = normalize_funcs
        self.normalize_literals = normalize_literals
        self.normalize_types = normalize_types
        self.preserve_keywords = preserve_keywords
        self.preserve_stdlib = preserve_stdlib
        self.tokenizer = Tokenizer()
        
    def _tokenize_with_types(self, code: str) -> List[Token]:
        """Tokenize code and return Token objects"""
        raw_tokens = self.tokenizer.tokenize_with_types(code)
        tokens = []
        pos = 0
        for value, type_str in raw_tokens:
            token_type = self._map_token_type(type_str)
            tokens.append(Token(value=value, type=token_type, position=pos))
            pos += len(value)
        return tokens
    
    def _map_token_type(self, type_str: str) -> TokenType:
        """Map string token type to TokenType enum"""
        mapping = {
            'KEYWORD': TokenType.KEYWORD,
            'IDENTIFIER': TokenType.IDENTIFIER,
            'NUMBER': TokenType.NUMBER,
            'HEX': TokenType.HEX,
            'STRING': TokenType.STRING,
            'CHAR': TokenType.CHAR,
            'OPERATOR': TokenType.OPERATOR,
            'PUNCTUATION': TokenType.PUNCTUATION,
            'COMMENT': TokenType.COMMENT,
            'WHITESPACE': TokenType.WHITESPACE,
        }
        return mapping.get(type_str, TokenType.UNKNOWN)
    
    def normalize_tokens(self, tokens: List[Token],
                        existing_maps: Optional[NormalizationMaps] = None
                        ) -> Tuple[List[str], NormalizationMaps]:
        """
        Normalize token sequence.
        Returns (normalized_tokens, maps)
        """
        maps = existing_maps.copy() if existing_maps else NormalizationMaps()
        normalized = []
        
        for i, token in enumerate(tokens):
            prev_token = tokens[i - 1] if i > 0 else None
            next_token = tokens[i + 1] if i < len(tokens) - 1 else None
            
            norm_value = self._normalize_token(token, maps, prev_token, next_token)
            normalized.append(norm_value)
        
        return normalized, maps
    
    def normalize_code(self, code: str,
                      existing_maps: Optional[NormalizationMaps] = None
                      ) -> Tuple[str, NormalizationMaps]:
        """
        Normalize raw code string.
        Tokenize -> normalize -> reconstruct
        """
        tokens = self._tokenize_with_types(code)
        normalized_tokens, maps = self.normalize_tokens(tokens, existing_maps)
        normalized_code = ' '.join(normalized_tokens)
        return normalized_code, maps
    
    def _normalize_token(self, token: Token, maps: NormalizationMaps,
                        prev_token: Optional[Token] = None,
                        next_token: Optional[Token] = None) -> str:
        """
        Normalize a single token based on its type and context.
        """
        value = token.value
        
        if token.type == TokenType.KEYWORD:
            return value
        
        if token.type == TokenType.STRING:
            if self.normalize_literals:
                return NORM_STR
            return value
        
        if token.type == TokenType.CHAR:
            if self.normalize_literals:
                return NORM_CHAR
            return value
        
        if token.type in (TokenType.NUMBER, TokenType.HEX):
            if self.normalize_literals:
                return self._normalize_literal(token)
            return value
        
        if token.type == TokenType.IDENTIFIER:
            if self.preserve_keywords and value in C_KEYWORDS:
                return value
            
            if value in C_TYPES:
                if self.normalize_types:
                    return maps.get_or_add_type(value)
                return value
            
            if self.preserve_stdlib and value in COMMON_STDLIB_FUNCS:
                return value
            
            is_func = self._is_function_call_context(prev_token, next_token)
            is_func_def = self._is_function_definition_context(prev_token, next_token)
            
            if is_func or is_func_def:
                if self.normalize_funcs:
                    return maps.get_or_add_func(value)
                return value
            else:
                if self.normalize_vars:
                    return maps.get_or_add_var(value)
                return value
        
        return value
    
    def _is_function_call_context(self, prev_token: Optional[Token],
                                   next_token: Optional[Token]) -> bool:
        """Detect if identifier is used as function (followed by '(')"""
        if next_token and next_token.value == '(':
            if prev_token and prev_token.value in ('.', '->'):
                return False
            return True
        return False
    
    def _is_function_definition_context(self, prev_token: Optional[Token],
                                         next_token: Optional[Token]) -> bool:
        """Detect function definition: type name(...)"""
        if next_token and next_token.value == '(':
            if prev_token:
                if prev_token.value in C_TYPES:
                    return True
                if prev_token.value == '*':
                    return True
                if prev_token.type == TokenType.IDENTIFIER:
                    if prev_token.value.endswith('_t') or prev_token.value in C_TYPES:
                        return True
        return False
    
    def _normalize_literal(self, token: Token) -> str:
        """Normalize literals: numbers -> NUM, strings -> STR"""
        value = token.value
        
        if token.type == TokenType.HEX:
            return NORM_NUM
        
        if token.type == TokenType.NUMBER:
            if value.lower().startswith('0x'):
                return NORM_NUM
            if '.' in value or value.lower().endswith(('f', 'l')):
                return NORM_FLOAT
            return NORM_NUM
        
        if token.type == TokenType.STRING:
            return NORM_STR
        
        if token.type == TokenType.CHAR:
            return NORM_CHAR
        
        return value
    
    def align_with_dataset(self, normalized_tokens: List[str],
                          dataset_normalized: str) -> List[str]:
        """
        Try to align our normalization with dataset's normalized_func.
        Useful to ensure consistency with Devign's existing normalization.
        """
        dataset_tokens = dataset_normalized.split()
        
        if len(normalized_tokens) != len(dataset_tokens):
            return normalized_tokens
        
        var_remap = {}
        func_remap = {}
        
        for our_tok, ds_tok in zip(normalized_tokens, dataset_tokens):
            if our_tok.startswith('VAR_') and ds_tok.startswith('VAR_'):
                var_remap[our_tok] = ds_tok
            elif our_tok.startswith('FUNC_') and ds_tok.startswith('FUNC_'):
                func_remap[our_tok] = ds_tok
        
        aligned = []
        for tok in normalized_tokens:
            if tok in var_remap:
                aligned.append(var_remap[tok])
            elif tok in func_remap:
                aligned.append(func_remap[tok])
            else:
                aligned.append(tok)
        
        return aligned


def _normalize_single(code: str, normalizer: CodeNormalizer) -> Tuple[List[str], NormalizationMaps]:
    """Helper for multiprocessing"""
    tokens = normalizer._tokenize_with_types(code)
    return normalizer.normalize_tokens(tokens)


def normalize_batch(codes: List[str], n_jobs: int = 4) -> List[Tuple[List[str], NormalizationMaps]]:
    """Batch normalization with multiprocessing"""
    normalizer = CodeNormalizer()
    
    if n_jobs == 1:
        return [_normalize_single(code, normalizer) for code in codes]
    
    results = []
    for code in codes:
        results.append(_normalize_single(code, normalizer))
    
    return results


def extract_normalization_from_dataset(original: str, normalized: str) -> NormalizationMaps:
    """
    Extract normalization mapping by comparing original and normalized code.
    Used to learn dataset's normalization pattern.
    """
    maps = NormalizationMaps()
    
    tokenizer = Tokenizer()
    orig_tokens = tokenizer.tokenize(original)
    norm_tokens = normalized.split()
    
    if len(orig_tokens) != len(norm_tokens):
        min_len = min(len(orig_tokens), len(norm_tokens))
        orig_tokens = orig_tokens[:min_len]
        norm_tokens = norm_tokens[:min_len]
    
    for orig, norm in zip(orig_tokens, norm_tokens):
        if norm.startswith('VAR_'):
            if orig not in maps.var_map:
                maps.var_map[orig] = norm
                idx = int(norm.split('_')[1])
                maps.var_counter = max(maps.var_counter, idx + 1)
        
        elif norm.startswith('FUNC_'):
            if orig not in maps.func_map:
                maps.func_map[orig] = norm
                idx = int(norm.split('_')[1])
                maps.func_counter = max(maps.func_counter, idx + 1)
        
        elif norm.startswith('TYPE_'):
            if orig not in maps.type_map:
                maps.type_map[orig] = norm
                idx = int(norm.split('_')[1])
                maps.type_counter = max(maps.type_counter, idx + 1)
    
    return maps


Normalizer = CodeNormalizer


def normalize_code(code: str) -> str:
    """Convenience function to normalize code"""
    normalizer = CodeNormalizer()
    normalized, _ = normalizer.normalize_code(code)
    return normalized


if __name__ == '__main__':
    test_code = '''
    int calculate_sum(int a, int b) {
        int result = a + b;
        printf("Sum: %d", result);
        return result;
    }
    '''
    
    normalizer = CodeNormalizer()
    normalized, maps = normalizer.normalize_code(test_code)
    
    print("Original:")
    print(test_code)
    print("\nNormalized:")
    print(normalized)
    print("\nMappings:")
    print(f"Variables: {maps.var_map}")
    print(f"Functions: {maps.func_map}")
    print(f"Reverse: {maps.reverse_map()}")
