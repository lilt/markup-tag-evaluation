#!/usr/bin/env python3

import argparse
from typing import List

from markup_tag_evaluation.tag_evaluation import evaluate_segments
from markup_tag_evaluation.parse_tags import extract_positions, extract_positions_v2


def parse_args():
    parser = argparse.ArgumentParser("Calculate the number of correctly placed tags (defined as <.*>)"
                                     " based on a reference and a hypothesis.")
    parser.add_argument("reference", help="Path of the reference file including tags.")
    parser.add_argument("hypothesis", help="Path of the hypothesis file including tags.")
    parser.add_argument("--permissive", action="store_true",
                        help="Do not throw errors for inconsistent reference and hypothesis sentences")
    parser.add_argument("--use_v2_tag_format", action="store_true",
                        help="Use version 2 tag format that allows arbitrary tag contents tag values")

    return parser.parse_args()


def read_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [" ".join(s.split()) for s in f]


if __name__ == "__main__":
    args = parse_args()
    reference = read_text(args.reference)
    hypothesis = read_text(args.hypothesis)

    if args.use_v2_tag_format:
        extract_tags_fn = extract_positions_v2
    else:
        extract_tags_fn = extract_positions

    tag_metric = evaluate_segments(reference, hypothesis, extract_tags_fn, args.permissive)
    print(tag_metric)
