#!/bin/bash

set -e

pycodestyle markup_tag_evaluation --max-line-length 100
mypy markup_tag_evaluation scripts/
python -m doctest markup_tag_evaluation/*.py

