from __future__ import annotations
from typing import List, NamedTuple, Tuple

import markup_tag_evaluation.constants as constants


class Tag(NamedTuple):
    content: str
    position: int


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
    split_by_tags = sentence_with_tags.split("<")
    sentence, tag_opening_splits = split_by_tags[0], split_by_tags[1:]

    for tag_content_to_next_tag in tag_opening_splits:
        # if tag is open, it has to be closed somewhere
        tag_content, string_after_tag = tag_content_to_next_tag.split(">")

        tags.append(Tag(tag_content, len(sentence)))
        sentence += string_after_tag

    return sentence, tags


def extract_positions_v2(sentence_with_tags: str) -> Tuple[str, List[Tag]]:

    tags: List[Tag] = []
    content_start_pos_stack: List[str] = []
    last_tag_was_tag_open_start = False
    last_tag_was_self_closing_open_start = False

    split_by_tags = sentence_with_tags.split(constants.V2_COMMON_TAG_PREFIX)

    sentence, tag_opening_splits = split_by_tags[0], split_by_tags[1:]

    for tag_content_to_next_tag in tag_opening_splits:
        tag_end_char_pos = tag_content_to_next_tag.find(">") + 1
        assert tag_end_char_pos
        current_tag = constants.V2_COMMON_TAG_PREFIX + tag_content_to_next_tag[:tag_end_char_pos]
        content = tag_content_to_next_tag[tag_end_char_pos:]

        if current_tag == constants.V2_TAG_OPEN_START:
            if last_tag_was_tag_open_start:
                raise ValueError(f"Inconsistent tag structure, "
                                 f"{constants.V2_TAG_OPEN_START} was not followed by "
                                 f"{constants.V2_TAG_OPEN_END}: {sentence_with_tags}")
            last_tag_was_tag_open_start = True
            content_start_pos_stack.append(content)
            new_tag = Tag(content, len(sentence))
            tags.append(new_tag)
            content = ""
        elif current_tag == constants.V2_TAG_OPEN_END:
            if not last_tag_was_tag_open_start:
                raise ValueError(f"Inconsistent tag structure, "
                                 f"{constants.V2_TAG_OPEN_START} was not followed by "
                                 f"{constants.V2_TAG_OPEN_END}: {sentence_with_tags}")
            last_tag_was_tag_open_start = False
        elif current_tag == constants.V2_TAG_CLOSE:
            if len(content_start_pos_stack) == 0:
                raise ValueError(f"Inconsistent tag structure, no opened tag to close with "
                                 f"{constants.V2_TAG_CLOSE}: {sentence_with_tags}")
            tag_content = "/" + content_start_pos_stack.pop()
            new_tag = Tag(tag_content, len(sentence))
            tags.append(new_tag)
        elif current_tag == constants.V2_SELF_CLOSING_START:
            if last_tag_was_tag_open_start:
                raise ValueError(f"Inconsistent tag structure for self closing tags:"
                                 f"{sentence_with_tags}")

            last_tag_was_self_closing_open_start = True
            new_tag = Tag(content, len(sentence))
            tags.append(new_tag)
            content = ""
        elif current_tag == constants.V2_SELF_CLOSING_END:
            assert last_tag_was_self_closing_open_start
            last_tag_was_self_closing_open_start = False
        else:
            raise ValueError(f"Unknown tag {current_tag}")

        # Inefficient as multiple string concatenations
        sentence += content

    if last_tag_was_tag_open_start or last_tag_was_self_closing_open_start \
            or len(content_start_pos_stack) > 0:
        raise ValueError(f"Inconsistent tag structure, not all tags were closed: "
                         f"{sentence_with_tags}")

    return sentence, tags
