import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.eval.genomic_evaluator import GenomicEvaluator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run genomic model evaluation.")
    parser.add_argument("--config", type=Path, required=True, help="Path to evaluation YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if "runs" not in cfg:
        raise SystemExit("Config must contain 'runs'.")

    evaluator = GenomicEvaluator()
    evaluator.run(args.config)


if __name__ == "__main__":
    main()
