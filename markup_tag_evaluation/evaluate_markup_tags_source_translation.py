#!/usr/bin/env python3

import argparse
from collections import Counter
from typing import List

from markup_tag_evaluation.parse_tags import extract_positions


def parse_args():
    parser = argparse.ArgumentParser("Check if the tag structure is consistent and that all tags "
                                     "from the source are present in the translation.")
    parser.add_argument("source", help="Path of the reference file including tags.")
    parser.add_argument("translation", help="Path of the hypothesis file including tags.")

    return parser.parse_args()


def read_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [s for s in f]


def clean_translation(segment: str) -> str:
    cnt_greater = segment.count(">")
    cnt_smaller = segment.count("<")
    # there should not be unmatched <> in the translation, try to clean them
    if cnt_greater != cnt_smaller and cnt_greater <= 1 and cnt_smaller <= 1:
        return segment.replace("<", "").replace(">", "")
    else:
        return segment


def main() -> None:
    args = parse_args()
    source_lines = read_text(args.source)
    translation_lines = read_text(args.translation)

    invalid_tag_structure = 0
    sentences_with_tags = 0
    number_of_tags = 0
    number_of_unmatched_tags = 0

    for src, trans in zip(source_lines, translation_lines, strict=True):
        _, src_tags = extract_positions(src)
        if len(src_tags) > 0:
            sentences_with_tags += 1
            number_of_tags += len(src_tags)

        try:
            _, trans_tags = extract_positions(clean_translation(trans))
        except ValueError:
            print(f"Invalid:\n{src=}\n{trans=}\n")
            invalid_tag_structure += 1
            number_of_unmatched_tags += len(src_tags)
            continue

        src_tags_counter = Counter(x.content for x in src_tags)
        trans_tags_counter = Counter(x.content for x in trans_tags)
        missing_tags_counter = src_tags_counter - trans_tags_counter
        assert min(missing_tags_counter.values(), default=0) >= 0
        number_of_unmatched_tags += sum(missing_tags_counter.values())
    print(f"Inconsistent tag structure: {invalid_tag_structure/sentences_with_tags:.1%} "
          f"({invalid_tag_structure}/{sentences_with_tags})")
    print(f"Unmatched tags: {number_of_unmatched_tags/number_of_tags:.1%} "
          f"({number_of_unmatched_tags}/{number_of_tags})")


if __name__ == "__main__":
    main()
