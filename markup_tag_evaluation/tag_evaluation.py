#!/usr/bin/env python3

from __future__ import annotations
from collections import defaultdict, Counter
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass

from markup_tag_evaluation.parse_tags import Tag


@dataclass
class TagMetric:
    number_of_tags: int
    number_of_correct_tags: int
    character_difference: int
    number_of_inconsistent_sentences: int
    number_of_tags_in_inconsistent_sentences: int
    number_of_sentences: int

    def accuracy(self) -> float:
        return self.number_of_correct_tags / max(1, self.number_of_tags)

    def average_character_difference(self) -> float:
        return self.character_difference / max(1, self.number_of_tags)

    def inconsistent_sentences_percentage(self) -> float:
        return self.number_of_inconsistent_sentences / max(1, self.number_of_sentences)

    def __add__(self, other: TagMetric) -> TagMetric:
        return TagMetric(
            self.number_of_tags + other.number_of_tags,
            self.number_of_correct_tags + other.number_of_correct_tags,
            self.character_difference + other.character_difference,
            self.number_of_inconsistent_sentences + other.number_of_inconsistent_sentences,
            self.number_of_tags_in_inconsistent_sentences +
            other.number_of_tags_in_inconsistent_sentences,
            self.number_of_sentences + other.number_of_sentences,
        )

    def __str__(self):
        acc_str = f"Tag Accuracy {self.accuracy():.1%} " \
                  f"({self.number_of_correct_tags}/{self.number_of_tags})"
        char_diff_str = f"Average character difference {self.average_character_difference():.1f} " \
                        f"({self.character_difference}/{self.number_of_tags})"
        result_array = [
            acc_str,
            char_diff_str,
        ]
        if self.number_of_inconsistent_sentences > 0:
            result_array.append(
                f"Inconsistent sentences {self.inconsistent_sentences_percentage():.1%} "
                f"({self.number_of_inconsistent_sentences}/{self.number_of_sentences})\n"
                f"Tags in inconsistent sentences: {self.number_of_tags_in_inconsistent_sentences}"
            )
        return "\n".join(result_array)


def create_inconsistent_tag_metric(number_of_tags_in_sentence: int) -> TagMetric:
    return TagMetric(
        number_of_tags=0,
        number_of_correct_tags=0,
        character_difference=0,
        number_of_inconsistent_sentences=1,
        number_of_tags_in_inconsistent_sentences=number_of_tags_in_sentence,
        number_of_sentences=1,
    )


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


def evaluate_segment(
        reference_with_tags: str,
        hypothesis_with_tags: str,
        tag_extraction_function: Callable[[str], Tuple[str, List[Tag]]],
        permissive: bool,
) -> TagMetric:
    try:
        ref_sentence, ref_tags = tag_extraction_function(reference_with_tags)
    except Exception as e:
        if permissive:
            print(f"Inconsistent reference, ignoring sentence: {e}")
            return TagMetric(0, 0, 0, 0, 0, 0)
        else:
            raise e

    try:
        hyp_sentence, hyp_tags = tag_extraction_function(hypothesis_with_tags)
    except Exception as e:
        if permissive:
            print(f"Inconsistent hypothesis: {e}")
            return create_inconsistent_tag_metric(len(ref_tags))
        else:
            raise e

    # Check for inconsistencies between reference and hypothesis
    counter_reference_tags = Counter(x.content for x in ref_tags)
    counter_hypothesis_tags = Counter(x.content for x in hyp_tags)
    error_message: Optional[str] = None
    if ref_sentence != hyp_sentence:
        strip_equal = ref_sentence.strip() == hyp_sentence.strip()
        error_message = (f"Reference without tags does not match hypothesis without tags: "
                         f"{ref_sentence=} {hyp_sentence=} {strip_equal=}")

    if error_message is None and counter_reference_tags != counter_hypothesis_tags:
        error_message = (f"Inconsistent number of tags between reference and hypothesis, "
                         f"{counter_reference_tags=} {counter_hypothesis_tags=}")
    if error_message is not None:
        if permissive:
            print(error_message)
            return create_inconsistent_tag_metric(len(ref_tags))
        else:
            raise ValueError(error_message)

    result = TagMetric(
        number_of_tags=len(ref_tags),
        number_of_correct_tags=tag_position_matches(ref_tags, hyp_tags),
        character_difference=position_differences(ref_tags, hyp_tags),
        number_of_inconsistent_sentences=0,
        number_of_tags_in_inconsistent_sentences=0,
        number_of_sentences=1,
    )
    return result


def evaluate_segments(
        reference_with_tags_list: List[str],
        hypothesis_with_tags_list: List[str],
        tag_extraction_function: Callable[[str], Tuple[str, List[Tag]]],
        permissive: bool,
) -> TagMetric:
    if len(reference_with_tags_list) != len(hypothesis_with_tags_list):
        raise ValueError(f"Inconsistent length of arguments: {len(reference_with_tags_list)=} "
                         f"{len(hypothesis_with_tags_list)=}")

    tag_metric_zero = TagMetric(0, 0, 0, 0, 0, 0)
    result = sum((evaluate_segment(ref, hyp, tag_extraction_function, permissive) for ref, hyp in
                  zip(reference_with_tags_list, hypothesis_with_tags_list)),
                 start=tag_metric_zero)
    return result
