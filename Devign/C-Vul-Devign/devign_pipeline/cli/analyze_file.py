"""CLI for running C vulnerability detection on a single file with localization."""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HierarchicalBiGRU vulnerability detection on a C file."
    )
    parser.add_argument("--file", "-f", required=True, help="Path to C source file.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON.",
    )
    parser.add_argument(
        "--github-annotations",
        action="store_true",
        help="Emit GitHub Actions annotations.",
    )
    parser.add_argument(
        "--localize",
        action="store_true",
        default=True,
        help="Enable attention-based vulnerability localization (default: True).",
    )
    parser.add_argument(
        "--no-localize",
        action="store_true",
        help="Disable localization, only show binary prediction.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of lines to highlight (default: 5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the default vulnerability threshold.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model file (default: models/best_v2_seed42.pt).",
    )
    parser.add_argument(
        "--path-prefix",
        type=str,
        default="",
        help="Prefix to add to file path in GitHub annotations (e.g., 'Devign/C-Vul-Devign/').",
    )
    return parser.parse_args()


def emit_github_annotation(
    path: str, 
    line: int,
    level: str,
    message: str,
) -> None:
    """Emit a GitHub Actions annotation."""
    print(f"::{level} file={path},line={line}::{message}")


def main() -> None:
    args = parse_args()
    file_path = Path(args.file)

    if not file_path.exists():
        error = {"error": f"File not found: {file_path}"}
        print(json.dumps(error), file=sys.stderr)
        sys.exit(2)

    if file_path.suffix.lower() not in (".c", ".h", ".cpp", ".cc", ".cxx"):
        error = {"error": f"Not a C/C++ file: {file_path}"}
        print(json.dumps(error), file=sys.stderr)
        sys.exit(2)

    code = file_path.read_text(encoding="utf-8", errors="ignore")

    from devign_pipeline.api.inference import get_model_wrapper, ModelWrapper

    if args.model_path:
        wrapper = ModelWrapper(model_paths=[Path(args.model_path)])
    else:
        wrapper = get_model_wrapper()

    if args.threshold is not None:
        wrapper.threshold = args.threshold

    use_localization = args.localize and not args.no_localize

    if use_localization:
        result = wrapper.predict_with_localization(code, top_k=args.top_k)
        prediction = result.prediction
        highlights = result.highlights
    else:
        prediction = wrapper.predict(code)
        highlights = []

    if args.github_annotations:
        annotation_path = args.path_prefix + str(file_path)
        if prediction.vulnerable:
            emit_github_annotation(
                path=annotation_path,
                line=1,
                level="error",
                message=f"VULNERABLE - Score: {prediction.score:.4f}, Confidence: {prediction.confidence}",
            )
            
            for i, loc in enumerate(highlights):
                emit_github_annotation(
                    path=annotation_path,
                    line=loc.line,
                    level="warning",
                    message=f"[Rank {i+1}] Suspicious code (attention={loc.normalized_score:.2f}): {loc.code_snippet[:60]}...",
                )
        else:
            emit_github_annotation(
                path=annotation_path,
                line=1,
                level="notice",
                message=f"Clean - Score: {prediction.score:.4f}, Confidence: {prediction.confidence}",
            )

    if args.json or not args.github_annotations:
        output = {
            "file": str(file_path),
            "vulnerable": prediction.vulnerable,
            "score": prediction.score,
            "threshold": prediction.threshold,
            "confidence": prediction.confidence,
        }
        
        if highlights:
            output["highlights"] = [
                {
                    "line": loc.line,
                    "score": round(loc.score, 6),
                    "normalized_score": round(loc.normalized_score, 4),
                    "code_snippet": loc.code_snippet,
                    "tokens": loc.tokens,
                }
                for loc in highlights
            ]
        
        print(json.dumps(output, indent=2))
    
    elif not args.github_annotations:
        status = "❌ VULNERABLE" if prediction.vulnerable else "✅ Clean"
        print(f"\n{status}")
        print(f"Score: {prediction.score:.4f} (threshold: {prediction.threshold:.2f})")
        print(f"Confidence: {prediction.confidence}")
        
        if highlights:
            print(f"\n⚠️  Top {len(highlights)} suspicious locations:")
            for i, loc in enumerate(highlights):
                print(f"\n  [{i+1}] Line {loc.line} (attention: {loc.normalized_score:.2f})")
                print(f"      {loc.code_snippet[:80]}")
                if loc.tokens:
                    print(f"      Tokens: {', '.join(loc.tokens[:5])}")

    if prediction.vulnerable:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
