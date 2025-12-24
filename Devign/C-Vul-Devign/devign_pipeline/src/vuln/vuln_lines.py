"""Shared utility for extracting vulnerability line numbers."""
from typing import Any, List
import numpy as np


def extract_vul_line_numbers(vul_lines_raw: Any) -> List[int]:
    """
    Extract vulnerability line numbers from vul_lines data.
    
    Handles multiple formats:
    - {'line_no': [41, 45, 47], 'code': [...]}  (Devign format)
    - {'line_no': np.array([41, 45, 47]), ...}  (numpy array)
    - [41, 45, 47]  (direct list)
    - {41: 'code1', 45: 'code2'}  (line -> code mapping)
    """
    if vul_lines_raw is None:
        return []
    
    if isinstance(vul_lines_raw, dict):
        if 'line_no' in vul_lines_raw:
            line_nos = vul_lines_raw['line_no']
            if hasattr(line_nos, 'tolist'):
                line_nos = line_nos.tolist()
            if isinstance(line_nos, (list, tuple)):
                return [int(l) for l in line_nos if isinstance(l, (int, float, np.integer))]
        else:
            keys = list(vul_lines_raw.keys())
            return [int(k) for k in keys if str(k).isdigit()]
    
    if isinstance(vul_lines_raw, (list, tuple)):
        return [int(l) for l in vul_lines_raw if isinstance(l, (int, float, np.integer))]
    
    if hasattr(vul_lines_raw, 'tolist'):
        return [int(l) for l in vul_lines_raw.tolist()]
    
    return []
