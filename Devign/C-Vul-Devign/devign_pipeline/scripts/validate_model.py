"""Validate model performance meets thresholds."""

import argparse
import json
import sys
from pathlib import Path


ENSEMBLE_CONFIG_PATH = Path("ensemble_config.json")

F1_THRESHOLD = 0.74
AUC_THRESHOLD = 0.87


def load_ensemble_config() -> dict:
    if ENSEMBLE_CONFIG_PATH.exists():
        return json.loads(ENSEMBLE_CONFIG_PATH.read_text())
    return {"optimal_threshold": 0.65}


def validate_from_config(config: dict, f1_min: float, auc_min: float) -> bool:
    print("=" * 60)
    print("Model Validation Report (from ensemble_config.json)")
    print("=" * 60)
    
    test_f1 = config.get("test_opt_f1", config.get("test_f1_05", 0))
    test_auc = config.get("test_auc", 0)
    test_precision = config.get("test_precision", 0)
    test_recall = config.get("test_recall", 0)
    threshold = config.get("optimal_threshold", 0.65)
    
    print(f"\nThreshold: {threshold:.4f}")
    print(f"\nMetrics:")
    print(f"  F1 Score:  {test_f1:.4f} (threshold: {f1_min})")
    print(f"  AUC-ROC:   {test_auc:.4f} (threshold: {auc_min})")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    
    passed = True
    print("\nValidation Results:")
    
    if test_f1 >= f1_min:
        print(f"  [PASS] F1 >= {f1_min}")
    else:
        print(f"  [FAIL] F1 >= {f1_min}")
        passed = False
    
    if test_auc >= auc_min:
        print(f"  [PASS] AUC >= {auc_min}")
    else:
        print(f"  [FAIL] AUC >= {auc_min}")
        passed = False
    
    print("=" * 60)
    
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model performance")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/best_v2_seed42.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=F1_THRESHOLD,
        help=f"Minimum F1 score (default: {F1_THRESHOLD})",
    )
    parser.add_argument(
        "--auc-threshold",
        type=float,
        default=AUC_THRESHOLD,
        help=f"Minimum AUC score (default: {AUC_THRESHOLD})",
    )
    args = parser.parse_args()
    
    config = load_ensemble_config()
    
    if args.model_path.exists():
        print(f"Model file found: {args.model_path}")
    else:
        print(f"Model file not found: {args.model_path}")
        print("Validating using ensemble_config.json metrics only...")
    
    passed = validate_from_config(config, args.f1_threshold, args.auc_threshold)
    
    if passed:
        print("\n[SUCCESS] MODEL VALIDATION PASSED")
        sys.exit(0)
    else:
        print("\n[FAILED] MODEL VALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
