#!/usr/bin/env python3

import argparse
from collections import defaultdict
from typing import List, Optional

from markup_tag_evaluation.tag_evaluation import TagMetric, TagMetrics, evaluate_segments
from markup_tag_evaluation.parse_tags import extract_positions, extract_positions_v2


def parse_args():
    parser = argparse.ArgumentParser("Calculate the number of correctly placed tags"
                                     " based on a reference and a hypothesis.")
    parser.add_argument("reference", help="Path of the reference file including tags.")
    parser.add_argument("hypothesis", help="Path of the hypothesis file including tags.")
    parser.add_argument("--source_lang", help="Path to file containing source language.")
    parser.add_argument("--target_lang", help="Path to file containing target language.")
    parser.add_argument("--csvout",
                        help="Path where to store a CSV with results aggregated by language.")
    parser.add_argument("--permissive", action="store_true",
                        help="Do not throw errors for inconsistent reference and hypothesis tags")
    parser.add_argument("--compare_strip", action="store_true",
                        help="Compare stripped strings without tags")
    parser.add_argument("--use_v2_tag_format", action="store_true",
                        help="Use version 2 tag format that allows arbitrary tag contents \
                            tag values")
    parser.add_argument("--backup_hypothesis",
                        help="Backup hypothesis in case the original "
                             "hypothesis has inconsistent tags")

    return parser.parse_args()


def read_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [s for s in f]


def main() -> None:
    args = parse_args()
    reference = read_text(args.reference)
    hypothesis = read_text(args.hypothesis)

    backup_hypothesis = None
    if args.backup_hypothesis is not None:
        backup_hypothesis: Optional[List[str]] = read_text(args.backup_hypothesis)

    source_lang = None
    if args.source_lang:
        source_lang = read_text(args.source_lang)

    target_lang = None
    if args.target_lang:
        target_lang = read_text(args.target_lang)

    if args.use_v2_tag_format:
        extract_tags_fn = extract_positions_v2
    else:
        extract_tags_fn = extract_positions

    tag_metrics = evaluate_segments(
        reference_with_tags_list=reference,
        hypothesis_with_tags_list=hypothesis,
        tag_extraction_function=extract_tags_fn,
        permissive=args.permissive,
        compare_strip=args.compare_strip,
        source_lang=source_lang,
        target_lang=target_lang,
        backup_hypothesis_with_tags_list=backup_hypothesis,
    )
    metrics_by_language = defaultdict(list)
    for tag_metric in tag_metrics:
        metrics_by_language[(tag_metric.src_language, tag_metric.tgt_language)].append(tag_metric)

    all_metrics = TagMetrics([])
    for language_pair, metrics in metrics_by_language.items():
        sum_language = sum(metrics, start=TagMetric.create_empty(src_language=language_pair[0], tgt_language=language_pair[1]))
        print(sum_language)
        all_metrics.metrics.append(sum_language)

    sum_all = sum(tag_metrics, start=TagMetric.create_empty(src_language="ALL", tgt_language="ALL"))
    print(sum_all)
    all_metrics.metrics.append(sum_all)

    if args.csvout:
        print(f"Writing CSV with results to {args.csvout}")
        all_metrics.to_csv(args.csvout)


if __name__ == "__main__":
    main()
