"""Generate SARIF output for GitHub Code Scanning integration."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SARIF report from vulnerability scan results."
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        required=True,
        help="List of C source files to scan.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results.sarif",
        help="Output SARIF file path.",
    )
    parser.add_argument(
        "--path-prefix",
        type=str,
        default="",
        help="Prefix to add to file paths.",
    )
    return parser.parse_args()


def create_sarif_template() -> Dict[str, Any]:
    """Create base SARIF structure."""
    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Devign Vulnerability Scanner",
                        "version": "2.0.0",
                        "informationUri": "https://github.com/tuananhquadeptrai/Degivn_V",
                        "rules": [
                            {
                                "id": "VULN001",
                                "name": "PotentialVulnerability",
                                "shortDescription": {
                                    "text": "Potential security vulnerability detected"
                                },
                                "fullDescription": {
                                    "text": "The BiGRU model detected patterns commonly associated with security vulnerabilities such as buffer overflows, use-after-free, format string bugs, etc."
                                },
                                "defaultConfiguration": {
                                    "level": "error"
                                },
                                "properties": {
                                    "tags": ["security", "vulnerability"],
                                    "precision": "medium",
                                    "security-severity": "7.0"
                                }
                            },
                            {
                                "id": "VULN002",
                                "name": "SuspiciousCodePattern",
                                "shortDescription": {
                                    "text": "Suspicious code pattern detected"
                                },
                                "fullDescription": {
                                    "text": "Attention-based localization identified this code region as potentially vulnerable."
                                },
                                "defaultConfiguration": {
                                    "level": "warning"
                                },
                                "properties": {
                                    "tags": ["security", "suspicious"],
                                    "precision": "low",
                                    "security-severity": "5.0"
                                }
                            }
                        ]
                    }
                },
                "results": []
            }
        ]
    }


def scan_file(file_path: str, path_prefix: str) -> List[Dict[str, Any]]:
    """Scan a single file and return SARIF results."""
    results = []
    
    path = Path(file_path)
    if not path.exists():
        return results
    
    if path.suffix.lower() not in (".c", ".h", ".cpp", ".cc", ".cxx"):
        return results
    
    code = path.read_text(encoding="utf-8", errors="ignore")
    
    # Import model wrapper
    from devign_pipeline.api.inference import get_model_wrapper
    wrapper = get_model_wrapper()
    
    # Get prediction with localization
    result = wrapper.predict_with_localization(code, top_k=5)
    prediction = result.prediction
    highlights = result.highlights
    
    artifact_path = path_prefix + str(file_path)
    
    if prediction.vulnerable:
        # Get the most suspicious line (highest attention) for main alert
        main_line = highlights[0].line if highlights else 1
        main_snippet = highlights[0].code_snippet if highlights else ""
        
        # Main vulnerability result - point to most suspicious line
        main_result = {
            "ruleId": "VULN001",
            "level": "error",
            "message": {
                "text": f"Potential vulnerability detected (Score: {prediction.score:.4f}, Confidence: {prediction.confidence}). Most suspicious: {main_snippet[:60]}"
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": artifact_path
                        },
                        "region": {
                            "startLine": main_line,
                            "snippet": {
                                "text": main_snippet
                            }
                        }
                    }
                }
            ],
            "properties": {
                "score": prediction.score,
                "threshold": prediction.threshold,
                "confidence": prediction.confidence,
                "detected_patterns": prediction.detected_patterns
            }
        }
        results.append(main_result)
        
        # Add each highlighted location as separate alert
        for i, loc in enumerate(highlights):
            loc_result = {
                "ruleId": "VULN002",
                "level": "warning",
                "message": {
                    "text": f"[Rank {i+1}] Suspicious code (attention={loc.normalized_score:.2f}): {loc.code_snippet[:80]}"
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": artifact_path
                            },
                            "region": {
                                "startLine": loc.line,
                                "snippet": {
                                    "text": loc.code_snippet
                                }
                            }
                        }
                    }
                ],
                "properties": {
                    "rank": i + 1,
                    "attention_score": loc.normalized_score,
                    "tokens": loc.tokens[:10]
                }
            }
            results.append(loc_result)
    
    return results


def main() -> None:
    args = parse_args()
    
    sarif = create_sarif_template()
    
    for file_path in args.files:
        print(f"Scanning: {file_path}", file=sys.stderr)
        results = scan_file(file_path, args.path_prefix)
        sarif["runs"][0]["results"].extend(results)
    
    # Write SARIF output
    output_path = Path(args.output)
    output_path.write_text(json.dumps(sarif, indent=2))
    
    total_vulns = len([r for r in sarif["runs"][0]["results"] if r["ruleId"] == "VULN001"])
    print(f"SARIF report written to: {args.output}", file=sys.stderr)
    print(f"Total vulnerabilities found: {total_vulns}", file=sys.stderr)
    
    # Exit with error if vulnerabilities found
    if total_vulns > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
