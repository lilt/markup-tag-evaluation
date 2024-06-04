#!/usr/bin/env python3

from __future__ import annotations
from collections import defaultdict, Counter
from typing import List, Tuple, NamedTuple, Optional
from dataclasses import dataclass


class Tag(NamedTuple):
    content: str
    position: int


@dataclass
class TagMetric:
    number_of_tags: int
    number_of_correct_tags: int
    character_difference: int
    number_of_inconsistent_sentences: int
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
            self.number_of_sentences + other.number_of_sentences,
        )

    def __str__(self):
        acc_str = f"Tag Accuracy {self.accuracy():.1%} " \
                  f"({self.number_of_correct_tags}/{self.number_of_tags})"
        char_diff_str = f"Average Character difference {self.average_character_difference():.1f} " \
                        f"({self.character_difference}/{self.number_of_tags})"
        result_array = [
            acc_str,
            char_diff_str,
        ]
        if self.number_of_inconsistent_sentences > 0:
            result_array.append(
                f"Inconsistent Sentences {self.inconsistent_sentences_percentage():.1%} "
                f"({self.number_of_inconsistent_sentences}/{self.number_of_sentences})"
            )
        return "\n".join(result_array)


SENTENCE_INCONSISTENT_TAG_METRIC = TagMetric(
    number_of_tags=0,
    number_of_correct_tags=0,
    character_difference=0,
    number_of_inconsistent_sentences=1,
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
        assert ref_tag.content in h_tags_dict
        diff = min(abs(ref_tag.position - h.position) for h in h_tags_dict[ref_tag.content])
        position_diff_sum += diff

    return position_diff_sum


def evaluate_segment(
        reference_with_tags: str,
        hypothesis_with_tags: str,
        permissive: bool,
) -> TagMetric:
    ref_sentence, ref_tags = extract_positions(reference_with_tags)
    try:
        hyp_sentence, hyp_tags = extract_positions(hypothesis_with_tags)
    except ValueError as e:
        if permissive:
            print(e)
            return SENTENCE_INCONSISTENT_TAG_METRIC
        else:
            raise e

    # Check for inconsistencies between reference and hypothesis
    counter_reference_tags = Counter(x.content for x in ref_tags)
    counter_hypothesis_tags = Counter(x.content for x in hyp_tags)
    error_message: Optional[str] = None
    if ref_sentence != hyp_sentence:
        error_message = (f"Reference without tags does not match hypothesis without tags: "
                         f"{ref_sentence=} {hyp_sentence=}")

    if error_message is None and counter_reference_tags != counter_hypothesis_tags:
        error_message = (f"Inconsistent number of tags between reference and hypothesis, "
                         f"{counter_reference_tags=} {counter_hypothesis_tags=}")
    if error_message is not None:
        if permissive:
            print(error_message)
            return SENTENCE_INCONSISTENT_TAG_METRIC
        else:
            raise ValueError(error_message)

    result = TagMetric(
        number_of_tags=len(ref_tags),
        number_of_correct_tags=tag_position_matches(ref_tags, hyp_tags),
        character_difference=position_differences(ref_tags, hyp_tags),
        number_of_inconsistent_sentences=0,
        number_of_sentences=1,
    )
    return result


def evaluate_segments(
        reference_with_tags_list: List[str],
        hypothesis_with_tags_list: List[str],
        permissive: bool,
) -> TagMetric:
    if len(reference_with_tags_list) != len(hypothesis_with_tags_list):
        raise ValueError(f"Inconsistent length of arguments: {len(reference_with_tags_list)=} "
                         f"{len(hypothesis_with_tags_list)=}")

    tag_metric_zero = TagMetric(0, 0, 0, 0, 0)
    result = sum((evaluate_segment(ref, hyp, permissive) for ref, hyp in
                  zip(reference_with_tags_list, hypothesis_with_tags_list)),
                 start=tag_metric_zero)
    return result
