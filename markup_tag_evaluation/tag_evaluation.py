#!/usr/bin/env python3

from __future__ import annotations
from collections import defaultdict, Counter
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass


class Tag(NamedTuple):
    content: str
    position: int


@dataclass
class TagMetric:
    number_of_tags: int
    number_of_correct_tags: int
    character_difference: int

    def accuracy(self) -> float:
        return self.number_of_correct_tags / max(1, self.number_of_tags)

    def average_character_difference(self) -> float:
        return self.character_difference / max(1, self.number_of_tags)

    def __add__(self, other: TagMetric) -> TagMetric:
        return TagMetric(
            self.number_of_tags + other.number_of_tags,
            self.number_of_correct_tags + other.number_of_correct_tags,
            self.character_difference + other.character_difference
        )

    def __str__(self):
        acc_str = f"{100.0 * self.accuracy():.1f}% " \
                  f"({self.number_of_correct_tags}/{self.number_of_tags})"
        char_diff_str = f"{self.average_character_difference():.1f} " \
                        f"({self.character_difference}/{self.number_of_tags})"

        return f"Tag Accuracy {acc_str}\nAverage Character Difference: {char_diff_str}"


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


def extract_positions(sentence_with_tags: str) -> Tuple[str, List[Tag]]:
    """ Returns the tuple (sentence, position_to_tags)
      sentence: sentence without tags
      position_to_tags: mapping of positon to an array of tags
    >>> extract_positions(" Hello world ")
    (' Hello world ', [])
    >>> extract_positions("<b>Hello</b> world!")
    ('Hello world!', [Tag(content='b', position=0), Tag(content='/b', position=5)])
    """
    if not is_sentence_with_tags_valid(sentence_with_tags):
        raise ValueError(f"Invalid tag structure: {sentence_with_tags=}")

    tags: List[Tag] = []

    # First array element will have no tags in it, e.g "Hello <1> there." => ["Hello ", "1> there."]
    splitted_by_tags = sentence_with_tags.split("<")
    sentence, tag_opening_splits = splitted_by_tags[0], splitted_by_tags[1:]

    for tag_content_to_next_tag in tag_opening_splits:
        # if tag is open, it has to be closed somewhere
        tag_content, string_after_tag = tag_content_to_next_tag.split(">")

        tags.append(Tag(tag_content, len(sentence)))
        sentence += string_after_tag

    return sentence, tags


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
        if ref_tag.content not in h_tags_dict:
            error_message = (f"Tag {ref_tag} does not appear in hypothesis tags list:"
                             f" {hypothesis_tags}.")
            raise ValueError(error_message)

        diff = min(abs(ref_tag.position - h.position) for h in h_tags_dict[ref_tag.content])
        position_diff_sum += diff

    return position_diff_sum


def evaluate_segment(reference_with_tags: str, hypothesis_with_tags: str) -> TagMetric:
    ref_sentence, ref_tags = extract_positions(reference_with_tags)
    hyp_sentence, hyp_tags = extract_positions(hypothesis_with_tags)

    if ref_sentence != hyp_sentence:
        error_message = (f"Reference without tags does not match hypothesis without tags: "
                         f"{ref_sentence=} {hyp_sentence=}")
        raise ValueError(error_message)

    result = TagMetric(
        number_of_tags=len(ref_tags),
        number_of_correct_tags=tag_position_matches(ref_tags, hyp_tags),
        character_difference=position_differences(ref_tags, hyp_tags),
    )
    return result


def evaluate_segments(
        reference_with_tags_list: List[str],
        hypothesis_with_tags_list: List[str]
) -> TagMetric:
    if len(reference_with_tags_list) != len(hypothesis_with_tags_list):
        raise ValueError(f"Inconsistent length of arguments: {len(reference_with_tags_list)=} "
                         f"{len(hypothesis_with_tags_list)=}")

    tag_metric_zero = TagMetric(0, 0, 0)
    result = sum((evaluate_segment(ref, hyp) for ref, hyp in
                  zip(reference_with_tags_list, hypothesis_with_tags_list)),
                 start=tag_metric_zero)
    return result
