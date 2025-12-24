"""Vulnerability dictionary and patterns"""

import re
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Pattern

logger = logging.getLogger(__name__)


@dataclass
class VulnerabilityPattern:
    """Represents a single vulnerability pattern"""
    name: str
    category: str  # buffer_overflow, null_pointer, use_after_free, integer_overflow, format_string
    pattern: str   # regex pattern
    description: str = ""
    severity: str = "medium"  # high, medium, low
    _compiled: Optional[Pattern[str]] = field(default=None, repr=False)
    
    def compile(self) -> Optional[Pattern[str]]:
        """Compile regex pattern with error handling"""
        if self._compiled is not None:
            return self._compiled
        try:
            self._compiled = re.compile(self.pattern, re.MULTILINE)
            return self._compiled
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{self.pattern}': {e}")
            return None
    
    def match(self, code: str) -> List[re.Match]:
        """Find all matches in code"""
        compiled = self.compile()
        if compiled is None:
            return []
        try:
            return list(compiled.finditer(code))
        except Exception as e:
            logger.warning(f"Error matching pattern '{self.name}': {e}")
            return []


class VulnDictionary:
    """Dictionary of vulnerability patterns with category management"""
    
    SEVERITY_WEIGHTS = {"high": 3, "medium": 2, "low": 1}
    DEFAULT_SEVERITY = {
        "buffer_overflow": "high",
        "use_after_free": "high",
        "null_pointer": "medium",
        "integer_overflow": "medium",
        "format_string": "high",
    }
    
    def __init__(self, patterns: Dict[str, Any] = None, config_path: str = None):
        self.raw_patterns = patterns or {}
        self.patterns: List[VulnerabilityPattern] = []
        self.categories: Dict[str, List[VulnerabilityPattern]] = {}
        self.dangerous_functions: Set[str] = set()
        self.vuln_types: List[str] = []
        
        if config_path:
            self.load_from_yaml(config_path)
        elif patterns:
            self._build_from_dict(patterns)
    
    def load_from_yaml(self, path: str) -> None:
        """Load patterns từ vuln_patterns.yaml"""
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.error(f"Config file not found: {path}")
                return
            
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                logger.warning(f"Empty config file: {path}")
                return
                
            self.raw_patterns = data
            self._build_from_dict(data)
            logger.info(f"Loaded {len(self.patterns)} patterns from {path}")
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parse error in {path}: {e}")
        except Exception as e:
            logger.error(f"Error loading config {path}: {e}")
    
    def _build_from_dict(self, data: Dict[str, Any]) -> None:
        """Build patterns and lookup tables from dictionary"""
        self.patterns = []
        self.categories = {}
        self.dangerous_functions = set()
        self.vuln_types = []
        
        for category, config in data.items():
            if not isinstance(config, dict):
                continue
                
            self.vuln_types.append(category)
            self.categories[category] = []
            
            description = config.get('description', f'{category} vulnerability')
            severity = self.DEFAULT_SEVERITY.get(category, "medium")
            
            funcs = config.get('dangerous_functions', [])
            if isinstance(funcs, list):
                self.dangerous_functions.update(funcs)
            
            regex_patterns = config.get('patterns', [])
            if not isinstance(regex_patterns, list):
                continue
                
            for i, pattern_str in enumerate(regex_patterns):
                if not isinstance(pattern_str, str) or not pattern_str.strip():
                    continue
                    
                vuln_pattern = VulnerabilityPattern(
                    name=f"{category}_{i}",
                    category=category,
                    pattern=pattern_str,
                    description=description,
                    severity=severity,
                )
                
                if vuln_pattern.compile() is not None:
                    self.patterns.append(vuln_pattern)
                    self.categories[category].append(vuln_pattern)
    
    def get_patterns_by_category(self, category: str) -> List[VulnerabilityPattern]:
        """Get all patterns for a specific category"""
        return self.categories.get(category, [])
    
    def get_all_dangerous_functions(self) -> List[str]:
        """Return list of all dangerous function names"""
        return sorted(list(self.dangerous_functions))
    
    def get_dangerous_functions_by_category(self, category: str) -> List[str]:
        """Get dangerous functions for a specific category"""
        config = self.raw_patterns.get(category, {})
        funcs = config.get('dangerous_functions', [])
        return funcs if isinstance(funcs, list) else []
    
    def get_vuln_type(self, function_name: str) -> List[str]:
        """Get vulnerability types for a function"""
        types = []
        for vuln_type, config in self.raw_patterns.items():
            if not isinstance(config, dict):
                continue
            if function_name in config.get('dangerous_functions', []):
                types.append(vuln_type)
        return types
    
    def is_dangerous(self, function_name: str) -> bool:
        """Check if function is dangerous"""
        return function_name in self.dangerous_functions
    
    def get_patterns(self, vuln_type: str) -> List[str]:
        """Get regex patterns (as strings) for vulnerability type"""
        return self.raw_patterns.get(vuln_type, {}).get('patterns', [])
    
    def get_severity(self, category: str) -> str:
        """Get severity level for a category"""
        return self.DEFAULT_SEVERITY.get(category, "medium")
    
    def get_severity_weight(self, category: str) -> int:
        """Get numeric severity weight for scoring"""
        severity = self.get_severity(category)
        return self.SEVERITY_WEIGHTS.get(severity, 2)
    
    def get_indicators(self, category: str) -> List[str]:
        """Get indicators for a vulnerability category"""
        config = self.raw_patterns.get(category, {})
        indicators = config.get('indicators', [])
        return indicators if isinstance(indicators, list) else []
    
    def __len__(self) -> int:
        return len(self.patterns)
    
    def __repr__(self) -> str:
        return f"VulnDictionary(categories={len(self.categories)}, patterns={len(self.patterns)})"


def load_vuln_patterns(path: str) -> VulnDictionary:
    """Load vulnerability patterns from YAML file"""
    return VulnDictionary(config_path=path)


def get_default_dictionary() -> VulnDictionary:
    """Get dictionary with default patterns (when no config file available)"""
    default_patterns = {
        "buffer_overflow": {
            "description": "Buffer overflow vulnerabilities",
            "dangerous_functions": ["strcpy", "strcat", "gets", "sprintf", "memcpy"],
            "patterns": [
                r"strcpy\s*\([^,]+,[^)]+\)",
                r"strcat\s*\([^,]+,[^)]+\)",
                r"gets\s*\([^)]+\)",
            ],
        },
        "format_string": {
            "description": "Format string vulnerabilities",
            "dangerous_functions": ["printf", "fprintf", "sprintf", "snprintf"],
            "patterns": [
                r'printf\s*\([^"]+\)',
            ],
        },
        "null_pointer": {
            "description": "Null pointer dereference",
            "patterns": [
                r"\*\s*\w+\s*(?:=|==)\s*NULL",
            ],
        },
        "use_after_free": {
            "description": "Use after free vulnerabilities",
            "patterns": [
                r"free\s*\([^)]+\)",
            ],
        },
        "integer_overflow": {
            "description": "Integer overflow vulnerabilities",
            "dangerous_functions": ["atoi", "atol", "strtol"],
            "patterns": [
                r"atoi\s*\([^)]+\)",
            ],
        },
    }
    return VulnDictionary(patterns=default_patterns)
