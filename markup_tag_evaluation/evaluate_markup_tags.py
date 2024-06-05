#!/usr/bin/env python3

import argparse
from typing import List

from markup_tag_evaluation.tag_evaluation import evaluate_segments


def parse_args():
    parser = argparse.ArgumentParser("Calculate the number of correctly placed tags"
                                     " based on a reference and a hypothesis.")
    parser.add_argument("reference", help="Path of the reference file including tags.")
    parser.add_argument("hypothesis", help="Path of the hypothesis file including tags.")
    parser.add_argument("--permissive", action="store_true",
                        help="Do not throw errors for inconsistent reference and hypothesis tags")

    return parser.parse_args()


def read_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [" ".join(s.split()) for s in f]


def main():
    args = parse_args()
    reference = read_text(args.reference)
    hypothesis = read_text(args.hypothesis)

    tag_metric = evaluate_segments(reference, hypothesis, args.permissive)
    print(tag_metric)


if __name__ == "__main__":
    main()
