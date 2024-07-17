#!/usr/bin/env python3

from __future__ import annotations
from collections import defaultdict, Counter
import csv
from enum import Enum
import re
from typing import List, Optional, Callable, Tuple, Union
from dataclasses import dataclass

from markup_tag_evaluation.parse_tags import Tag


class InconsistencyType(Enum):
    HYPOTHESIS = "HYPOTHESIS"
    TAG_COUNT = "TAG_COUNT"
    TEXT = "TEXT"


@dataclass
class TagMetric:
    num_ref_tags: int
    num_correct_tags: int
    character_difference: int

    num_inconsistent_hyp: int
    num_inconsistent_tag_count: int
    num_inconsistent_text: int

    num_tags_inconsistent_sentences: int
    num_sentences: int
    tgt_language: str

    @staticmethod
    def create_empty(tgt_language: str) -> TagMetric:
        return TagMetric(
            num_ref_tags=0,
            num_correct_tags=0,
            character_difference=0,
            num_inconsistent_hyp=0,
            num_inconsistent_tag_count=0,
            num_inconsistent_text=0,
            num_tags_inconsistent_sentences=0,
            num_sentences=0,
            tgt_language=tgt_language,
        )

    @staticmethod
    def create_inconsistent(
        number_of_tags_in_sentence: int,
        tgt_language: str,
        inconsistency_type: InconsistencyType,
    ) -> TagMetric:
        metric = TagMetric.create_empty(tgt_language=tgt_language)
        metric.num_sentences = 1
        metric.num_tags_inconsistent_sentences = number_of_tags_in_sentence
        if inconsistency_type is InconsistencyType.HYPOTHESIS:
            metric.num_inconsistent_hyp = 1
        elif inconsistency_type is InconsistencyType.TAG_COUNT:
            metric.num_inconsistent_tag_count = 1
        elif inconsistency_type is InconsistencyType.TEXT:
            metric.num_inconsistent_text = 1
        else:
            raise ValueError(f"Unsupported {inconsistency_type=}")

        return metric

    def __add__(self, other: TagMetric) -> TagMetric:
        return TagMetric(
            num_ref_tags=self.num_ref_tags + other.num_ref_tags,
            num_correct_tags=self.num_correct_tags + other.num_correct_tags,
            character_difference=self.character_difference + other.character_difference,
            num_inconsistent_hyp=self.num_inconsistent_hyp + other.num_inconsistent_hyp,
            num_inconsistent_tag_count=(self.num_inconsistent_tag_count +
                                        other.num_inconsistent_tag_count),
            num_inconsistent_text=self.num_inconsistent_text + other.num_inconsistent_text,
            num_tags_inconsistent_sentences=(self.num_tags_inconsistent_sentences +
                                             other.num_tags_inconsistent_sentences),
            num_sentences=self.num_sentences + other.num_sentences,
            tgt_language=self.tgt_language,
        )

    def __str__(self):
        acc_str = f"Tag Accuracy {self.accuracy():.1%} " \
                  f"({self.num_correct_tags}/{self.num_ref_tags})"
        char_diff_str = f"Average character difference {self.average_character_difference():.1f} " \
                        f"({self.character_difference}/{self.num_ref_tags})"
        result_array = [
            f"Language: {self.tgt_language}",
            acc_str,
            char_diff_str,
        ]
        if self.number_of_inconsistent_sentences() > 0:
            result_array.append(
                f"Inconsistent sentences {self.inconsistent_sentences_percentage():.1%} "
                f"({self.number_of_inconsistent_sentences()}/{self.num_sentences})\n"
                f"Tags in inconsistent sentences: {self.num_tags_inconsistent_sentences}"
            )
        return "\n".join(result_array)

    def accuracy(self) -> float:
        return self.num_correct_tags / max(1, self.num_ref_tags)

    def average_character_difference(self) -> float:
        return self.character_difference / max(1, self.num_ref_tags)

    def number_of_inconsistent_sentences(self) -> int:
        return (self.num_inconsistent_hyp +
                self.num_inconsistent_tag_count +
                self.num_inconsistent_text)

    def inconsistent_sentences_percentage(self) -> float:
        return self.number_of_inconsistent_sentences() / max(1, self.num_sentences)

    def to_dict(self) -> dict:
        return {
            "tgt_language": self.tgt_language,
            "num_sentences": self.num_sentences,
            "accuracy": self.accuracy(),
            "num_correct_tags": self.num_correct_tags,
            "num_ref_tags": self.num_ref_tags,
            "average_character_difference": self.average_character_difference(),
            "num_inconsistent_total": self.number_of_inconsistent_sentences(),
            "num_inconsistent_hyp": self.num_inconsistent_hyp,
            "num_inconsistent_tag_count": self.num_inconsistent_tag_count,
            "num_inconsistent_text": self.num_inconsistent_text,
            "num_tags_inconsistent_sentences": self.num_tags_inconsistent_sentences,
        }


@dataclass
class TagMetrics:
    metrics: list[TagMetric]

    def to_csv(self, csv_filename):
        dict_rows = [metric.to_dict() for metric in self.metrics]
        if len(dict_rows) > 0:
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=dict_rows[0].keys())
                writer.writeheader()
                writer.writerows(dict_rows)


def is_sentence_with_tags_valid(sentence: str) -> bool:
    """ Check if a sentence contains valid tags and no other appearances of '<' and '>'
    >>> is_sentence_with_tags_valid("Hello")
    True
    >>> is_sentence_with_tags_valid("<tag> <tag> <tag>")
    True
    >>> is_sentence_with_tags_valid("<<tag>>")
    False
    >>> is_sentence_with_tags_valid("<tag> < <tag>")
    False
    """
    stack: List[str] = []
    for c in sentence:
        if c == '<':
            stack.append(c)
        elif c == '>':
            if len(stack) == 1:
                stack.pop()
            else:
                return False
    return len(stack) == 0


def tag_position_matches(reference_tags: List[Tag], hypothesis_tags: List[Tag]) -> int:
    """
    Returns the number of tags that are at the same character position
    as their matching tags in the reference.
    >>> tag_position_matches([Tag("a", 1)], [Tag("a", 1)])
    1
    >>> tag_position_matches([Tag("a", 1), Tag("a", 5)], [Tag("a", 1), Tag("a", 1)])
    1
    """
    r_tag_counter = Counter(reference_tags)
    h_tag_counter = Counter(hypothesis_tags)
    diff_counter = r_tag_counter - h_tag_counter

    incorrect = sum(diff_counter.values())
    correct = len(reference_tags) - incorrect

    return correct


def potentially_expand_self_closing_tags(
        hyp_tags: List[Tag],
        reference_tags_counter: Counter[str],
) -> List[Tag]:
    hyp_tags_counter = Counter(t.content for t in hyp_tags)
    for hyp_t in hyp_tags:
        closing_counterpart_str = "/" + hyp_t.content
        if (closing_counterpart_str not in hyp_tags_counter
                and closing_counterpart_str in reference_tags_counter):
            new_closing_tag = Tag(closing_counterpart_str, hyp_t.position)
            hyp_tags.append(new_closing_tag)
            hyp_tags_counter[closing_counterpart_str] += 1
    return hyp_tags


def position_differences(reference_tags: List[Tag], hypothesis_tags: List[Tag]) -> int:
    """ Returns the sum character difference between matching tags in the reference and hypothesis.
        Is generous and selects the closest hypothesis tag with the same content
        in case of ambiguity.

    >>> position_differences([Tag("a", 2), Tag("b", 0)], [Tag("a", 2), Tag("b", 5)])
    5
    >>> position_differences([Tag("a", 2), Tag("a", 0)], [Tag("a", 2), Tag("a", 5)])
    2
    """
    position_diff_sum = 0
    h_tags_dict = defaultdict(list)
    for t in hypothesis_tags:
        h_tags_dict[t.content].append(t)

    for ref_tag in reference_tags:
        assert ref_tag.content in h_tags_dict
        diff = min(abs(ref_tag.position - h.position) for h in h_tags_dict[ref_tag.content])
        position_diff_sum += diff

    return position_diff_sum


_EXTRACT_LANG_REGEX = re.compile("<LANGUAGE_(..)>")


def extract_language(text: str) -> str:
    matches = _EXTRACT_LANG_REGEX.findall(text)
    if len(matches) > 0:
        return matches[0]
    else:
        raise ValueError(f"Failed to extract language from: {text}")


_DEFAULT_LANG = "UNK"


def evaluate_segment(
        source_with_tags: Optional[str],
        reference_with_tags: str,
        hypothesis_with_tags: str,
        tag_extraction_function: Callable[[str], Tuple[str, List[Tag]]],
        permissive: bool,
        compare_strip: bool = False,
) -> TagMetric:
    tgt_language = _DEFAULT_LANG
    if source_with_tags is not None:
        try:
            tgt_language = extract_language(source_with_tags)
        except Exception as e:
            if permissive:
                print(f"Cannot extract language from source: {source_with_tags}")
            else:
                raise e

    try:
        ref_sentence, ref_tags = tag_extraction_function(reference_with_tags)
    except Exception as e:
        if permissive:
            print(f"Inconsistent reference, ignoring sentence: {e}")
            return TagMetric.create_empty(tgt_language=tgt_language)
        else:
            raise e

    try:
        hyp_sentence, hyp_tags = tag_extraction_function(hypothesis_with_tags)
    except Exception as e:
        if permissive:
            print(f"Inconsistent hypothesis: {e}")
            return TagMetric.create_inconsistent(
                number_of_tags_in_sentence=len(ref_tags),
                tgt_language=tgt_language,
                inconsistency_type=InconsistencyType.HYPOTHESIS,
            )
        else:
            raise e

    # Check for inconsistencies between reference and hypothesis
    counter_reference_tags = Counter(x.content for x in ref_tags)
    if permissive:
        hyp_tags = potentially_expand_self_closing_tags(
            hyp_tags, counter_reference_tags
        )
    counter_hypothesis_tags = Counter(x.content for x in hyp_tags)

    if counter_reference_tags != counter_hypothesis_tags:
        error_message = (f"Inconsistent number of tags between reference and hypothesis, "
                         f"{counter_reference_tags=} {counter_hypothesis_tags=} "
                         f"{reference_with_tags=} {hypothesis_with_tags=}")
        if permissive:
            print(error_message)
            return TagMetric.create_inconsistent(
                number_of_tags_in_sentence=len(ref_tags),
                tgt_language=tgt_language,
                inconsistency_type=InconsistencyType.TAG_COUNT,
            )
        else:
            raise ValueError(error_message)

    if compare_strip:
        ref_sentence = ref_sentence.strip()
        hyp_sentence = hyp_sentence.strip()
    if ref_sentence != hyp_sentence:
        strip_equal = ref_sentence.strip() == hyp_sentence.strip()
        error_message = (f"Reference without tags does not match hypothesis without tags: "
                         f"{ref_sentence=} {hyp_sentence=} {strip_equal=}")
        if permissive:
            print(error_message)
            return TagMetric.create_inconsistent(
                number_of_tags_in_sentence=len(ref_tags),
                tgt_language=tgt_language,
                inconsistency_type=InconsistencyType.TEXT,
            )
        else:
            raise ValueError(error_message)

    result = TagMetric(
        num_ref_tags=len(ref_tags),
        num_correct_tags=tag_position_matches(ref_tags, hyp_tags),
        character_difference=position_differences(ref_tags, hyp_tags),
        num_inconsistent_hyp=0,
        num_inconsistent_tag_count=0,
        num_inconsistent_text=0,
        num_tags_inconsistent_sentences=0,
        num_sentences=1,
        tgt_language=tgt_language,
    )
    return result


def evaluate_segments(
        source_with_tags_list: Optional[List[str]],
        reference_with_tags_list: List[str],
        hypothesis_with_tags_list: List[str],
        tag_extraction_function: Callable[[str], Tuple[str, List[Tag]]],
        permissive: bool,
        compare_strip: bool = False,
        backup_hypothesis_with_tags_list: Optional[List[str]] = None,
) -> list[TagMetric]:
    if (len(reference_with_tags_list) != len(hypothesis_with_tags_list)) or \
       (source_with_tags_list is not None and
           len(source_with_tags_list) != len(reference_with_tags_list)):
        if source_with_tags_list is None:
            source_err = f"{source_with_tags_list=}"
        else:
            source_err = f"{len(source_with_tags_list)=}"
        raise ValueError(f"Inconsistent length of arguments: "
                         f"{len(reference_with_tags_list)=} "
                         f"{len(hypothesis_with_tags_list)=} "
                         f"{source_err}")
    if (backup_hypothesis_with_tags_list is not None and
            len(reference_with_tags_list) != len(backup_hypothesis_with_tags_list)):
        raise ValueError(f"Inconsistent length of reference and backup hypothesis "
                         f"{len(reference_with_tags_list)=} "
                         f"{len(backup_hypothesis_with_tags_list)=}")

    results = []
    for i, (ref, hyp) in enumerate(zip(reference_with_tags_list, hypothesis_with_tags_list)):
        src = source_with_tags_list[i] if source_with_tags_list is not None else None
        backup_hyp = backup_hypothesis_with_tags_list[i] \
            if backup_hypothesis_with_tags_list is not None else None

        permissive_first_run = permissive and backup_hyp is None
        try:
            res = evaluate_segment(src, ref, hyp, tag_extraction_function,
                                   permissive_first_run, compare_strip)
        except Exception as e:
            if backup_hyp is not None:
                res = evaluate_segment(src, ref, backup_hyp, tag_extraction_function,
                                       permissive, compare_strip)
            else:
                raise e
        results.append(res)
    return results
