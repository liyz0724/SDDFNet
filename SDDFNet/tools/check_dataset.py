"""Validate prepared image pairs and masks before running an experiment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.base_dataset import LabeledDataset
from datasets.supervised_dataset import collect_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    files = collect_files(args.roots, recursive=args.recursive)
    dataset = LabeledDataset(files, list(range(len(files))))
    for i in range(len(dataset)):
        dataset[i]
    print(json.dumps({"validated_pairs": len(dataset)}))


if __name__ == "__main__":
    main()
